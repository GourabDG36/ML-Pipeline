import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ML Pipeline",
  description: "Automated ML Classification Pipeline",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <nav className="bg-gray-900 text-white px-6 py-4 flex items-center gap-8">
          <span className="font-bold text-xl text-blue-400">ML Pipeline</span>
          <Link href="/" className="hover:text-blue-400 transition-colors">
            Upload
          </Link>
          <Link href="/train" className="hover:text-blue-400 transition-colors">
            Train
          </Link>
          <Link href="/predict" className="hover:text-blue-400 transition-colors">
            Predict
          </Link>
          <Link href="/drift" className="hover:text-blue-400 transition-colors">
            Drift Monitor
          </Link>
        </nav>
        <main className="min-h-screen bg-gray-950 text-white p-6">
          {children}
        </main>
      </body>
    </html>
  );
}