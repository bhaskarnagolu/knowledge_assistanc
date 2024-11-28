import React from 'react';
import { HStack, Card, Button } from "@chakra-ui/react";
import Image from 'next/image';

export default function ChatBox() {
  return (
    <HStack w="1146px" minW="250px" p={10} pb={0}>
      <Card
        flex={1}
        pl={5}
        pt={5}
        h="85px"
        bg="#ffffff"
        borderTopLeftRadius={40}
        borderTopRightRadius={40}
        borderBottomLeftRadius={0}
        borderBottomRightRadius={0}
      >
        <HStack>
          <HStack flex={1}>
            <input
              autoFocus
              type="text"
              onChange={(e) => console.log(e.target.value)}
              placeholder="Ask a question to start..."
              style={{
                height: '53px',
                width: '100%',
                backgroundColor: 'white',
                outline: 'none',
              }}
            />
          </HStack>
          <HStack pr={5}>
            <Button
              rightIcon={
                <Image
                  src="/send.svg"
                  alt="send"
                  width={25}
                  height={25}
                  priority
                />
              }
              color="#ffffff"
              colorScheme="messenger"
              borderRadius={40}
              opacity={1}
              fontWeight={400}
              variant="solid"
            >
              Start a new chat
            </Button>
          </HStack>
        </HStack>
      </Card>
    </HStack>
  );
}
