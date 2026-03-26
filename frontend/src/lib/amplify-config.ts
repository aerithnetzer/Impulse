"use client";

import { Amplify } from "aws-amplify";

Amplify.configure({
  Auth: {
    Cognito: {
      userPoolId: process.env.NEXT_PUBLIC_USER_POOL_ID || "",
      userPoolClientId: process.env.NEXT_PUBLIC_USER_POOL_CLIENT_ID || "",
      loginWith: {
        oauth: {
          domain: process.env.NEXT_PUBLIC_COGNITO_DOMAIN || "",
          scopes: ["openid", "email"],
          redirectSignIn: [
            process.env.NEXT_PUBLIC_REDIRECT_SIGN_IN ||
              "http://localhost:3000/dashboard/",
            "https://main.d3muotvhy8m9f1.amplifyapp.com/dashboard/",
          ],
          redirectSignOut: [
            process.env.NEXT_PUBLIC_REDIRECT_SIGN_OUT ||
              "http://localhost:3000/",
            "https://main.d3muotvhy8m9f1.amplifyapp.com/",
          ],
          responseType: "code",
        },
      },
    },
  },
});

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:3001";
