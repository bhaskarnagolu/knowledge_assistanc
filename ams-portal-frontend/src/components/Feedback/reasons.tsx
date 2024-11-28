import { IReasons } from '@/models';
import { HStack, Text } from "@chakra-ui/react";
import Image from 'next/image';
import React from 'react';

interface IReasonsProps {
  reasons: IReasons[];
  onHandleReason: React.Dispatch<React.SetStateAction<IReasons[]>>;
}
export default function Reasons({ reasons, onHandleReason }: IReasonsProps) {
  return (
    <HStack pt={3}>
      {reasons.map((el: IReasons, idx: number) => (
        <HStack
          key={idx}
          p={3}
          cursor="pointer"
          borderRadius={40}
          bg={el.isSelected ? '#EDF5FF' : undefined}
          border="1px solid lightgray"
          onClick={() =>
            onHandleReason((prev) =>
              prev.map((pEl: IReasons) => {
                if (pEl.title === el.title) pEl.isSelected = !pEl.isSelected;
                return pEl;
              })
            )
          }
        >
          {el.isSelected && (
            <Image
              src="/checkmark.svg"
              alt="checkmark"
              width={20}
              height={20}
              priority
            />
          )}
          <Text color="black">{el.title}</Text>
        </HStack>
      ))}
    </HStack>
  );
}
