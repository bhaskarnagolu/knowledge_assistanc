import { NextRequest, NextResponse } from "next/server";
import { FILE_KEY } from "@/lib/validations/upload.schema";
import { getErrorResponse } from "@/lib/helpers";

export async function POST(req: NextRequest) {
  const data = await req.formData();
  const file: File | null = data.get(FILE_KEY) as unknown as File;
  if (!file) {
    return getErrorResponse(400, "Invalid file input");
  }

  const bytes = await file.arrayBuffer()
  const buffer = Buffer.from(bytes)

  // Do something with the file!
  try {
    const sendResponse = await fetch(`http://www.hisapi.com/file`, {
    method: 'POST'
    });

    // The request succeeded - log out the response
  } catch (err: unknown) {
    // The request failed, we need to let user know
  }

  const response = new NextResponse(
    JSON.stringify({
      status: "success",
    }),
    {
      status: 201,
      headers: { "Content-Type": "application/json" },
    }
  );

  return response;
}