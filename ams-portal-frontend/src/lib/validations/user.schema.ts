import { z } from "zod";

export const LoginUserSchema = z.object({
  email: z
    .string({
      required_error: "Email is required",
    })
    .min(1, "Email is required")
    .email("Email is invalid"),
  password: z
    .string({
      required_error: "Password is required",
    })
    .min(1, "Password is required")
    .min(3, "Password must be at least 3 characters"),
});

export type LoginUserInput = z.infer<typeof LoginUserSchema>;
