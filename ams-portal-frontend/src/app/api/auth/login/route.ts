import { getEnvVariable, getErrorResponse } from "@/lib/helpers";
import { signJWT } from "@/lib/token";
import { LoginUserInput, LoginUserSchema } from "@/lib/validations/user.schema";
// import { compare } from "bcryptjs";
import { NextRequest, NextResponse } from "next/server";
import { ZodError } from "zod";
import { CognitoUserPool, CognitoUser, AuthenticationDetails, CognitoUserSession } from 'amazon-cognito-identity-js';

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as LoginUserInput;
    const data = LoginUserSchema.parse(body);

    const poolData = {
      UserPoolId: 'ap-southeast-2_fF69GbkQS',
      ClientId: 'cb9vanvn4i41pdrlo7170npsg',
    };

    const userPool = new CognitoUserPool(poolData);

    const authenticationData = {
      Username: body.email,
      Password: body.password,
    };

    const authenticationDetails = new AuthenticationDetails(authenticationData);

    const userData = {
      Username: body.email,
      Pool: userPool,
    };
    const cognitoUser = new CognitoUser(userData);
    
    let token = ''
    let idtoken = ''
    try {
      const session: CognitoUserSession = await new Promise((resolve, reject) => {
        cognitoUser.authenticateUser(authenticationDetails, {
          onSuccess: (session) => resolve(session),
          onFailure: (err) => reject(err),
        });
      });
      token = session.getAccessToken().getJwtToken();
      idtoken = session.getIdToken().getJwtToken();
    } catch (error) {
      console.log("Error: ", error)
      return getErrorResponse(
        401,
        "Incorrect username or password!"
      ); 
    }
    
    const cookieOptions = {
      name: "token",
      value: token,
      httpOnly: true,
      path: "/",
      secure: process.env.NODE_ENV !== "development",
    };

    const cookieOptions2 = {
      name: "idtoken",
      value: idtoken,
      httpOnly: true,
      path: "/",
      secure: process.env.NODE_ENV !== "development",
    };

    //const url = req.nextUrl.clone();
    //url.pathname = "/dashboard";

    const url = "https://main.dbuddmc5hbu78.amplifyapp.com/dashboard"

    const response = NextResponse.redirect(url.toString(), {status: 302});

    await Promise.all([
      response.cookies.set(cookieOptions),
      response.cookies.set(cookieOptions2),
      response.cookies.set({
        name: "redirectUrl",
        value: url.toString(),
        httpOnly: true,
        path: "/",
        secure: process.env.NODE_ENV !== "development"
      }),
      response.cookies.set({
        name: "redirectUrl",
        value: req.url.toString() || "",
        httpOnly: true,
        path: "/",
        secure: process.env.NODE_ENV !== "development"
      }),
      response.cookies.set({
        name: "logged-in",
        value: "true",
      }),
    ]);

    return response;

  } catch (error: any) {
    console.log("Error in catch: ", error)
    if (error instanceof ZodError) {
      return getErrorResponse(400, "failed validations", error);
    }

    return getErrorResponse(500, error.message);
  }
}