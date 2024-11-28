'use client';

import { Box, AbsoluteCenter, Spinner } from "@chakra-ui/react";

export default function loading() {
  return (
    <Box position="relative">
      <AbsoluteCenter p={4} axis="both">
        <Spinner
          thickness='4px'
          speed='0.65s'
          emptyColor='gray.200'
          color='blue.500'
          size='xl'
        />
      </AbsoluteCenter>
    </Box>
  );
}
