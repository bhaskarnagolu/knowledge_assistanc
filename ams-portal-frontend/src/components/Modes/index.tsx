import React from 'react';
import { HStack, Text } from "@chakra-ui/react";
import { IModeSelections } from '@/models';

interface IModes {
  selections: IModeSelections[];
  onChangeMode: React.Dispatch<React.SetStateAction<IModeSelections[]>>;
}
export default function Modes({ selections, onChangeMode }: IModes) {
  return (
    <HStack bg="#dde1e6" borderRadius={40} w="340px">
      {selections.map((el: IModeSelections, idx: number) => (
        <HStack key={idx} p={1}>
          {el.isSelected ? (
            <HStack
              bg="#ffffff"
              borderRadius={40}
              pl={3}
              pr={3}
              cursor="pointer"
            >
              {el.icon}
              <Text color="black">{el.title}</Text>
            </HStack>
          ) : (
            <HStack
              pl={3}
              pr={3}
              cursor="pointer"
              onClick={() =>
                onChangeMode((prev) =>
                  prev.map((pEl: IModeSelections) => {
                    pEl.title === el.title
                      ? (pEl.isSelected = true)
                      : (pEl.isSelected = false);

                    return pEl;
                  })
                )
              }
            >
              {el.icon}
              <Text color="black">{el.title}</Text>
            </HStack>
          )}
        </HStack>
      ))}
    </HStack>
  );
}
