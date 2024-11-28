import './globals.css';
import type { Metadata } from 'next';
import { IBM_Plex_Sans } from 'next/font/google';
import { Providers } from './providers';

// Load the IBM Plex Sans font from Google Fonts
const ibm = IBM_Plex_Sans({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
});

// Define metadata for the application
export const metadata: Metadata = {
  title: 'NextGen AMS Knowledge Assistant',
  description: 'NextGen AMS Knowledge Assistant - Self-Serve Portal',
};

// Define the RootLayout component for the application
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="theme-light">
      <head>
        {/* Metadata */}
        <title>{metadata.title}</title>
        <meta charSet="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta name="description" content={metadata.description} />
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body className={`${ibm.className} bg-gray-50 text-gray-900`}>
        {/* Top-Level Providers */}
        <Providers>
          {/* Header */}
          <header className="w-full py-4 bg-indigo-600 text-white shadow-md">
            <div className="container mx-auto flex justify-between items-center px-6">
              <h1 className="text-xl font-semibold">{metadata.title}</h1>
              <nav>
                <ul className="flex gap-4">
                  <li>
                    <a href="/" className="hover:underline">
                      Home
                    </a>
                  </li>
                  <li>
                    <a href="/about" className="hover:underline">
                      About
                    </a>
                  </li>
                  <li>
                    <a href="/contact" className="hover:underline">
                      Contact
                    </a>
                  </li>
                </ul>
              </nav>
            </div>
          </header>

          {/* Main Content */}
          <main className="container mx-auto p-6">{children}</main>

          {/* Footer */}
          <footer className="w-full py-4 bg-gray-800 text-white text-center">
            <p className="text-sm">
              © {new Date().getFullYear()} NextGen AMS Knowledge Assistant. All rights reserved.
            </p>
          </footer>
        </Providers>
      </body>
    </html>
  );
}
