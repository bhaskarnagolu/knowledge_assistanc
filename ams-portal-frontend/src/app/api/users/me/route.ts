import { getErrorResponse } from "@/lib/helpers";
import { NextRequest, NextResponse } from "next/server";
import { CognitoIdToken } from "amazon-cognito-identity-js";

export async function GET(req: NextRequest) {
  const idtoken = req.cookies.get('idtoken');

  if (!idtoken) {
    return getErrorResponse(
      401,
      "You are not logged in, please provide token to gain access"
    );
  }

  const userData = new CognitoIdToken({
    IdToken: idtoken.value
  }).decodePayload();

  if (!userData.email || !userData["custom:account_id"]) {
    return getErrorResponse(
      401,
      "You are not logged in, please provide token to gain access"
    );
  }

  const user = {
    email: userData.email,
    accountId: userData["custom:account_id"],
    name: userData.email
  }

  return NextResponse.json({
    status: "success",
    data: { user: { ...user, password: undefined } },
  });
}