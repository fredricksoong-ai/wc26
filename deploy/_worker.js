// Cloudflare Pages advanced-mode Worker (direct-upload safe).
// Handles POST /refresh -> triggers the GitHub Actions workflow_dispatch.
// Everything else falls through to the static site (env.ASSETS).
// Token lives ONLY here as a Cloudflare secret (GH_DISPATCH_TOKEN), set in
// Pages > wc26 > Settings > Variables and Secrets (type: Secret). The PAT needs
// fine-grained permission Actions = Read and write on fredricksoong-ai/wc26.

const REPO = "fredricksoong-ai/wc26";
const WORKFLOW = "predict.yml";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/refresh") {
      if (request.method !== "POST") return json({ ok: false, error: "POST only" }, 405);
      if (!env.GH_DISPATCH_TOKEN) return json({ ok: false, error: "GH_DISPATCH_TOKEN not configured" }, 500);
      const r = await fetch(`https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW}/dispatches`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${env.GH_DISPATCH_TOKEN}`,
          "Accept": "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
          "User-Agent": "wc26-dashboard",
        },
        body: JSON.stringify({ ref: "main" }),
      });
      if (r.status === 204) return json({ ok: true });
      return json({ ok: false, status: r.status, error: await r.text() }, 502);
    }
    return env.ASSETS.fetch(request);
  },
};

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json" } });
}
