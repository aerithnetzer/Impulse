import AnalysisWorkspacePage from "./client";

// Only needed for `next build` with output: export
export async function generateStaticParams() {
  return [{ id: "_" }];
}

export default function Page() {
  return <AnalysisWorkspacePage />;
}
