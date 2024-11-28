import React, { ReactNode } from 'react';
import { HStack, Card } from "@chakra-ui/react";

interface IOptions {
  children: ReactNode;
}
export default function Options({ children }: IOptions) {
  return (
    <HStack w="1146px" minW="250px" p={10} pt={0}>
      <Card
        flex={1}
        bg="#f3f4f8"
        h="40px"
        pl={10}
        pt={1}
        borderTopLeftRadius={0}
        borderTopRightRadius={0}
        borderBottomLeftRadius={40}
        borderBottomRightRadius={40}
      >
        {children}
      </Card>
    </HStack>
  );
}
