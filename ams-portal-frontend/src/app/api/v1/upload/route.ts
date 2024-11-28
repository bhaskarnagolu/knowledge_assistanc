import { getErrorResponse } from "@/lib/helpers";
import { NextRequest, NextResponse } from "next/server";

const API_HOST_URL = process.env.API_HOST_URL || "";

export async function POST(req: NextRequest) {

    try {
        const headers: Record<string, string> = {
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*',
            'Accept': '*/*',
            'X-Auth-Token': process.env.API_SECRET_TOKEN || ""
        };

        console.log(headers)
        const body_to_send = await req.formData();
        console.log(body_to_send);

        const response = await fetch(`${API_HOST_URL}/v1/uploadData`, {
            method: "PUT",
            body: body_to_send,
            headers: headers
        });

        const response_data = await response.json();
        console.log(response_data);

        return new NextResponse(
            JSON.stringify(response_data),
            {
                status: 200,
                headers: { "Content-Type": "application/json" },
            }
        );
    } catch (error: any) {
        return getErrorResponse(500, error.message);
    }
}