/** @type {import('next').NextConfig} */

// The API has no CORS middleware, and it does not need any: every browser request goes
// to this app's own origin at /api/*, and Next proxies it to FastAPI server-side. One
// origin means no preflight, no allow-list to keep in sync with deploy URLs, and no
// credentials exposed to the browser.
//
// Change API_URL in the environment to point at a deployed API; nothing in the app code
// knows the backend's address.
const API_URL = process.env.API_URL ?? "http://localhost:8000";

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_URL}/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;
