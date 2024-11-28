'use client';

import { Box, VStack, Heading } from "@chakra-ui/react";
import LoginForm from "./login-form";

export default async function LoginPage() {

  return (
    <Box bg="whiteAlpha.300" borderWidth="1px" borderRadius="lg" py={50} px={100}>
      <VStack spacing={5} alignItems="flex-start" flex={1}>
        <Heading as="h1" color="black" fontWeight="bold" fontSize="4xl">Welcome Back</Heading>
        <Heading as="h2" fontSize="2xl">Login to have access</Heading>
        <Box w="100%" py={8}>
          <LoginForm />
        </Box>
      </VStack>
    </Box>
  );
}
