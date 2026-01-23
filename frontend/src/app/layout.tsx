import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AI Trading Lab",
  description: "AI Portfolio Managers competing with Semantic Market Memory",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <div className="min-h-screen bg-background">
          <header className="border-b">
            <div className="container mx-auto px-4 py-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">🧪</span>
                  <div>
                    <h1 className="text-xl font-bold">AI Trading Lab</h1>
                    <p className="text-xs text-muted-foreground">
                      Do LLMs beat algorithms?
                    </p>
                  </div>
                </div>
                <nav className="flex items-center gap-4">
                  <a 
                    href="/" 
                    className="text-sm text-muted-foreground hover:text-foreground"
                  >
                    Dashboard
                  </a>
                  <a 
                    href="/managers" 
                    className="text-sm text-muted-foreground hover:text-foreground"
                  >
                    Managers
                  </a>
                  <a 
                    href="/signals" 
                    className="text-sm text-muted-foreground hover:text-foreground"
                  >
                    Signals
                  </a>
                </nav>
              </div>
            </div>
          </header>
          <main className="container mx-auto px-4 py-6">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
