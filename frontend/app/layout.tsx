import type React from "react"
import type { Metadata } from "next"
import { GeistSans } from "geist/font/sans"
import { Toaster } from "@/components/ui/toaster"

import "./globals.css"

export const metadata: Metadata = {
  title: "AI Document Search - Chat with your PDFs",
  description: "An intelligent RAG chatbot for PDF document search using LLM and semantic search",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={GeistSans.className}>
        {children}
        <Toaster />
      </body>
    </html>
  )
}



import './globals.css'