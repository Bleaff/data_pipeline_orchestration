import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "neudc control plane",
  description: "Read-only dashboard over the neudc pipeline control-plane API",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
