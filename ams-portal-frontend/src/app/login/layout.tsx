'use client';

import { useState } from "react";
import { HStack, VStack } from "@chakra-ui/react";

import { Layout } from '@/components';

export default function DashboardLayout({
  children
}: {
  children: React.ReactNode
}) {
  const [close, onClose] = useState(false);
  return (
    <Layout onClose={onClose} isOpen={close} isMainPage>
      <VStack w="100%">
        <HStack flex={1}>
          {children}
        </HStack>
      </VStack>
    </Layout>
  )
}