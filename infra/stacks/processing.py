"""Processing stack: Step Functions, ECS Fargate, Lambda."""

from pathlib import Path

from constructs import Construct
import aws_cdk as cdk
import aws_cdk.aws_ec2 as ec2
import aws_cdk.aws_ecr as ecr
import aws_cdk.aws_ecs as ecs
import aws_cdk.aws_iam as iam
import aws_cdk.aws_lambda as _lambda
import aws_cdk.aws_logs as logs
import aws_cdk.aws_s3 as s3
import aws_cdk.aws_sqs as sqs
import aws_cdk.aws_stepfunctions as sfn
import aws_cdk.aws_stepfunctions_tasks as sfn_tasks

_SRC_DIR = str(Path(__file__).resolve().parent.parent.parent / "src")


class ProcessingStack(cdk.Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        *,
        vpc: ec2.IVpc,
        bucket: s3.IBucket,
        image_repo: ecr.IRepository,
        extraction_repo: ecr.IRepository,
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)

        # ── ECS Cluster ──────────────────────────────────────────────────

        cluster = ecs.Cluster(
            self,
            "ImpulseCluster",
            vpc=vpc,
            cluster_name="impulse-processing",
            container_insights_v2=ecs.ContainerInsights.ENABLED,
        )

        # ── Dead Letter Queue ────────────────────────────────────────────

        dlq = sqs.Queue(
            self,
            "ProcessingDLQ",
            queue_name="impulse-processing-dlq",
            retention_period=cdk.Duration.days(14),
        )

        # ── Shared IAM policy for S3 + MongoDB Secrets ───────────────────

        task_policy = iam.PolicyStatement(
            actions=[
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket",
            ],
            resources=[
                bucket.bucket_arn,
                f"{bucket.bucket_arn}/*",
            ],
        )

        secrets_policy = iam.PolicyStatement(
            actions=["secretsmanager:GetSecretValue"],
            resources=[
                f"arn:aws:secretsmanager:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:secret:impulse/*",
            ],
        )

        ai_policy = iam.PolicyStatement(
            actions=[
                "bedrock:InvokeModel",
                "bedrock:ListFoundationModels",
                "bedrock:GetFoundationModel",
                "textract:DetectDocumentText",
                "textract:StartDocumentTextDetection",
                "textract:GetDocumentTextDetection",
                "aws-marketplace:ViewSubscriptions",
                "aws-marketplace:Subscribe",
            ],
            resources=[
                "*"
            ],  # Bedrock/Textract don't support resource-level restrictions
        )

        # ── ECS Task Definition: Image Processing ────────────────────────

        image_task_def = ecs.FargateTaskDefinition(
            self,
            "ImageProcessingTaskDef",
            memory_limit_mib=8192,
            cpu=2048,
            family="impulse-image-processing",
        )
        image_task_def.add_to_task_role_policy(task_policy)
        image_task_def.add_to_task_role_policy(secrets_policy)

        image_container = image_task_def.add_container(
            "ImageProcessor",
            image=ecs.ContainerImage.from_ecr_repository(image_repo, "latest"),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="image-processing",
                log_retention=logs.RetentionDays.ONE_MONTH,
            ),
            environment={
                "IMPULSE_BUCKET": bucket.bucket_name,
                "MONGODB_SECRET_ID": "impulse/mongodb-uri",
            },
        )

        # ── ECS Task Definition: Document Extraction ─────────────────────

        extraction_task_def = ecs.FargateTaskDefinition(
            self,
            "ExtractionTaskDef",
            memory_limit_mib=30720,  # 30 GB for Marker PDF
            cpu=4096,
            family="impulse-document-extraction",
        )
        extraction_task_def.add_to_task_role_policy(task_policy)
        extraction_task_def.add_to_task_role_policy(secrets_policy)

        extraction_container = extraction_task_def.add_container(
            "DocumentExtractor",
            image=ecs.ContainerImage.from_ecr_repository(extraction_repo, "latest"),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="document-extraction",
                log_retention=logs.RetentionDays.ONE_MONTH,
            ),
            environment={
                "IMPULSE_BUCKET": bucket.bucket_name,
                "MONGODB_SECRET_ID": "impulse/mongodb-uri",
            },
        )

        # ── Lambda: Lightweight tasks (METS, metadata, summaries, etc.) ──

        _bundling = cdk.BundlingOptions(
            image=_lambda.Runtime.PYTHON_3_13.bundling_image,
            command=[
                "bash",
                "-c",
                "pip install -r requirements-lambda.txt -t /asset-output"
                " && cp -au . /asset-output"
                " && rm -f /asset-output/requirements-lambda.txt",
            ],
        )

        # Separate bundling for the validator / processing Lambdas that
        # need pymupdf for PDF page expansion.
        _processing_bundling = cdk.BundlingOptions(
            image=_lambda.Runtime.PYTHON_3_13.bundling_image,
            command=[
                "bash",
                "-c",
                "pip install -r requirements-processing-lambda.txt -t /asset-output"
                " && cp -au . /asset-output"
                " && rm -f /asset-output/requirements-processing-lambda.txt",
            ],
        )

        lightweight_lambda = _lambda.Function(
            self,
            "LightweightProcessor",
            function_name="impulse-lightweight-processor",
            runtime=_lambda.Runtime.PYTHON_3_13,
            handler="impulse.handlers.lambda_handler.handler",
            code=_lambda.Code.from_asset(_SRC_DIR, bundling=_processing_bundling),
            memory_size=3008,
            timeout=cdk.Duration.minutes(15),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            environment={
                "IMPULSE_BUCKET": bucket.bucket_name,
                "MONGODB_SECRET_ID": "impulse/mongodb-uri",
                "BEDROCK_REGION": "us-east-1",
            },
        )
        lightweight_lambda.add_to_role_policy(task_policy)
        lightweight_lambda.add_to_role_policy(secrets_policy)
        lightweight_lambda.add_to_role_policy(ai_policy)
        bucket.grant_read_write(lightweight_lambda)

        # ── Lambda: Job Validator (enumerates pages, sets up work) ───────
        # The validator now expands PDF pages to images, so it needs
        # pymupdf, more memory, a longer timeout, and S3 write access.

        validator_lambda = _lambda.Function(
            self,
            "JobValidator",
            function_name="impulse-job-validator",
            runtime=_lambda.Runtime.PYTHON_3_13,
            handler="impulse.handlers.lambda_handler.handler",
            code=_lambda.Code.from_asset(_SRC_DIR, bundling=_processing_bundling),
            memory_size=3008,
            timeout=cdk.Duration.minutes(15),
            environment={
                "IMPULSE_BUCKET": bucket.bucket_name,
                "MONGODB_SECRET_ID": "impulse/mongodb-uri",
            },
        )
        validator_lambda.add_to_role_policy(task_policy)
        validator_lambda.add_to_role_policy(secrets_policy)
        bucket.grant_read_write(validator_lambda)

        # ── Step Functions: State Machine ────────────────────────────────

        # Step 1: Validate job & list documents
        validate_job = sfn_tasks.LambdaInvoke(
            self,
            "ValidateJob",
            lambda_function=validator_lambda,
            payload=sfn.TaskInput.from_object(
                {
                    "task_type": "validate",
                    "job_id.$": "$.job_id",
                    "input_s3_prefix.$": "$.input_s3_prefix",
                }
            ),
            result_path="$.validation",
            result_selector={
                "documents.$": "$.Payload.body.documents",
                "total.$": "$.Payload.body.total",
            },
        )

        # Step 2a: Run ECS Image Processing task
        run_image_processing = sfn_tasks.EcsRunTask(
            self,
            "RunImageProcessing",
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            cluster=cluster,
            task_definition=image_task_def,
            launch_target=sfn_tasks.EcsFargateLaunchTarget(
                platform_version=ecs.FargatePlatformVersion.LATEST,
            ),
            subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            container_overrides=[
                sfn_tasks.ContainerOverride(
                    container_definition=image_container,
                    environment=[
                        sfn_tasks.TaskEnvironmentVariable(
                            name="DOCUMENT_KEY",
                            value=sfn.JsonPath.string_at("$.document_key"),
                        ),
                        sfn_tasks.TaskEnvironmentVariable(
                            name="OUTPUT_KEY",
                            value=sfn.JsonPath.string_at("$.output_key"),
                        ),
                        sfn_tasks.TaskEnvironmentVariable(
                            name="JOB_ID",
                            value=sfn.JsonPath.string_at("$.job_id"),
                        ),
                    ],
                )
            ],
            result_path=sfn.JsonPath.DISCARD,
        )

        # Step 2b: Run ECS Document Extraction task
        run_extraction = sfn_tasks.EcsRunTask(
            self,
            "RunDocumentExtraction",
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            cluster=cluster,
            task_definition=extraction_task_def,
            launch_target=sfn_tasks.EcsFargateLaunchTarget(
                platform_version=ecs.FargatePlatformVersion.LATEST,
            ),
            subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            container_overrides=[
                sfn_tasks.ContainerOverride(
                    container_definition=extraction_container,
                    environment=[
                        sfn_tasks.TaskEnvironmentVariable(
                            name="DOCUMENT_KEY",
                            value=sfn.JsonPath.string_at("$.document_key"),
                        ),
                        sfn_tasks.TaskEnvironmentVariable(
                            name="JOB_ID",
                            value=sfn.JsonPath.string_at("$.job_id"),
                        ),
                        sfn_tasks.TaskEnvironmentVariable(
                            name="IMPULSE_IDENTIFIER",
                            value=sfn.JsonPath.string_at("$.impulse_identifier"),
                        ),
                    ],
                )
            ],
            result_path=sfn.JsonPath.DISCARD,
        )

        # Step 2c: Run Lambda for lightweight tasks
        run_lightweight = sfn_tasks.LambdaInvoke(
            self,
            "RunLightweightTask",
            lambda_function=lightweight_lambda,
            payload=sfn.TaskInput.from_json_path_at("$"),
            result_path=sfn.JsonPath.DISCARD,
        )

        # Task type router
        choose_task = (
            sfn.Choice(self, "ChooseTaskType")
            .when(
                sfn.Condition.string_equals("$.task_type", "image_transform"),
                run_image_processing,
            )
            .when(
                sfn.Condition.string_equals("$.task_type", "document_extraction"),
                run_extraction,
            )
            .otherwise(run_lightweight)
        )

        # Step 3: Fan-out over documents using Distributed Map
        process_documents = sfn.DistributedMap(
            self,
            "ProcessDocuments",
            items_path="$.validation.documents",
            max_concurrency=50,  # Control Fargate quota usage
            result_path=sfn.JsonPath.DISCARD,
        )
        process_documents.item_processor(choose_task)

        # Step 4: Mark job complete
        mark_complete = sfn_tasks.LambdaInvoke(
            self,
            "MarkJobComplete",
            lambda_function=validator_lambda,
            payload=sfn.TaskInput.from_object(
                {
                    "task_type": "complete",
                    "job_id.$": "$.job_id",
                }
            ),
            result_path=sfn.JsonPath.DISCARD,
        )

        # Step 5: Mark job failed (error handler)
        mark_failed = sfn_tasks.LambdaInvoke(
            self,
            "MarkJobFailed",
            lambda_function=validator_lambda,
            payload=sfn.TaskInput.from_object(
                {
                    "task_type": "fail",
                    "job_id.$": "$.job_id",
                    "error.$": "$.error_info.Error",
                    "cause.$": "$.error_info.Cause",
                }
            ),
            result_path=sfn.JsonPath.DISCARD,
        )
        # After marking failed, terminate the execution as failed
        pipeline_fail = sfn.Fail(
            self,
            "PipelineFailed",
            error="ProcessingFailed",
            cause="One or more documents failed to process",
        )
        mark_failed.next(pipeline_fail)

        # Wire it all together with error handling
        process_documents.add_catch(
            mark_failed,
            errors=["States.ALL"],
            result_path="$.error_info",
        )
        validate_job.add_catch(
            mark_failed,
            errors=["States.ALL"],
            result_path="$.error_info",
        )

        definition = validate_job.next(process_documents).next(mark_complete)

        self.state_machine = sfn.StateMachine(
            self,
            "ImpulseStateMachine",
            state_machine_name="impulse-document-pipeline",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            timeout=cdk.Duration.hours(48),
            logs=sfn.LogOptions(
                destination=logs.LogGroup(
                    self,
                    "StateMachineLog",
                    log_group_name="/aws/stepfunctions/impulse-pipeline",
                    retention=logs.RetentionDays.ONE_MONTH,
                ),
                level=sfn.LogLevel.ERROR,
            ),
        )

        # Grant the state machine permissions
        bucket.grant_read_write(self.state_machine)

        # ── Outputs ──────────────────────────────────────────────────────

        cdk.CfnOutput(
            self,
            "StateMachineArn",
            value=self.state_machine.state_machine_arn,
        )
        cdk.CfnOutput(
            self,
            "ClusterArn",
            value=cluster.cluster_arn,
        )
        cdk.CfnOutput(
            self,
            "DlqUrl",
            value=dlq.queue_url,
        )
