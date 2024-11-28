import { getErrorResponse } from "@/lib/helpers";
import { NextRequest, NextResponse } from "next/server";
import { parse } from "url";
import { ZodError } from "zod";

const API_HOST_URL = process.env.API_HOST_URL || "";

export async function OPTIONS(req: NextRequest) {

    try {
        const headers: Record<string, string> = {
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Accept': '*/*',
            'X-Auth-Token': process.env.API_SECRET_TOKEN || ""
        };

        console.log(headers)
        const { query } = parse(req.url, true); 

        console.log(query)

        const response = await fetch(`${API_HOST_URL}/v1/downloadResult?accountId=${query['accountId']}&jobId=${query['jobId']}`, {
            method: "GET",
            headers: headers
        });

        // const response_data = await response.json();
        // console.log(response_data);

        return new NextResponse(
            // return the blob in the response.body
            response.body,
            {
                status: 200,
                headers: { "Content-Type": "application/csv" },
            }
        );
    } catch (error: any) {
        return getErrorResponse(500, error.message);
        }
}