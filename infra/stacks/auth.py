"""Auth stack: Cognito User Pool for frontend authentication."""

from constructs import Construct
import aws_cdk as cdk
import aws_cdk.aws_cognito as cognito


class AuthStack(cdk.Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # ── User Pool ────────────────────────────────────────────────────

        self.user_pool = cognito.UserPool(
            self,
            "ImpulseUserPool",
            user_pool_name="impulse-users",
            self_sign_up_enabled=False,  # Admin-only user creation
            sign_in_aliases=cognito.SignInAliases(email=True),
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            password_policy=cognito.PasswordPolicy(
                min_length=12,
                require_lowercase=True,
                require_uppercase=True,
                require_digits=True,
                require_symbols=True,
            ),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            removal_policy=cdk.RemovalPolicy.RETAIN,
        )

        # ── App Client (for Next.js frontend) ───────────────────────────

        self.user_pool_client = self.user_pool.add_client(
            "ImpulseWebClient",
            user_pool_client_name="impulse-web",
            auth_flows=cognito.AuthFlow(
                user_srp=True,
                user_password=False,
            ),
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(
                    authorization_code_grant=True,
                ),
                scopes=[cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL],
                callback_urls=[
                    "http://localhost:3000/dashboard",
                    "http://localhost:3000/dashboard/",
                    "https://main.d3muotvhy8m9f1.amplifyapp.com/dashboard",
                    "https://main.d3muotvhy8m9f1.amplifyapp.com/dashboard/",
                ],
                logout_urls=[
                    "http://localhost:3000/",
                    "https://main.d3muotvhy8m9f1.amplifyapp.com/",
                ],
            ),
            generate_secret=False,  # Public client for SPA
            id_token_validity=cdk.Duration.hours(1),
            access_token_validity=cdk.Duration.hours(1),
            refresh_token_validity=cdk.Duration.days(30),
        )

        # ── Domain (for hosted UI) ──────────────────────────────────────

        self.user_pool_domain = self.user_pool.add_domain(
            "ImpulseCognitoDomain",
            cognito_domain=cognito.CognitoDomainOptions(
                domain_prefix="impulse-auth",
            ),
        )

        # ── Outputs ──────────────────────────────────────────────────────

        cdk.CfnOutput(self, "UserPoolId", value=self.user_pool.user_pool_id)
        cdk.CfnOutput(
            self,
            "UserPoolClientId",
            value=self.user_pool_client.user_pool_client_id,
        )
        cdk.CfnOutput(
            self,
            "CognitoDomain",
            value=self.user_pool_domain.base_url(),
        )
