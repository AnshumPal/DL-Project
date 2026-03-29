/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    unoptimized: true,
  },
  async rewrites() {
    // BACKEND_URL is a server-side env var (not exposed to the browser).
    // NEXT_PUBLIC_API_URL is the browser-side var used in fetch() calls.
    // Both default to http://localhost:8000.
    const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000"
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
      {
        source: "/health",
        destination: `${backendUrl}/health`,
      },
    ]
  },
}

export default nextConfig
