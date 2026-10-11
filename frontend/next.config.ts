import type { NextConfig } from "next";

// Static export: `npm run build` writes plain HTML/JS/CSS to `out/`, which
// `kernelsight serve` hosts next to the API. No Node server is needed at run time.
const nextConfig: NextConfig = {
  output: "export",
};

export default nextConfig;
