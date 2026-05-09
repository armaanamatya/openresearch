import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ReproLab Agent Lab",
  description: "Topology, activity, citations, and delegation detail views for ReproLab."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
