/**
 * Static export config — used for GitHub Pages deployment.
 * `rewrites()` is NOT supported in static export mode.
 * The frontend talks directly to the deployed backend via NEXT_PUBLIC_API_URL.
 *
 * Build with:
 *   NEXT_PUBLIC_API_URL=https://your-backend.onrender.com next build --config next.config.static.mjs
 */

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",       // emit static HTML/CSS/JS to /out
  trailingSlash: true,    // required for GitHub Pages routing
  images: {
    unoptimized: true,    // next/image optimization needs a server
  },
  // basePath must match the GitHub repo name when hosted at
  // https://<user>.github.io/<repo>/
  basePath: process.env.NEXT_PUBLIC_BASE_PATH ?? "",
}

export default nextConfig
