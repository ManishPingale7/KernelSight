import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "KernelSight",
  description: "PTX static analysis and compiler output comparison for CUDA kernels",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
