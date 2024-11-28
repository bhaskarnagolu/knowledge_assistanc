'use client';

import { useState } from "react";
import { Layout } from "@/components";
import { VStack, Text } from "@chakra-ui/react";

export default function Home() {
  const [close, onClose] = useState(true);

  return (
    <Layout onClose={onClose} isOpen={close} isMainPage={false}>
      <VStack py={10}>
        <Text>NextGen AMS Knowledge Assistant</Text>
      </VStack>
    </Layout>
  );
}
