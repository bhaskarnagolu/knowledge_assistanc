import React from 'react';
import { Box, FormLabel, FormErrorMessage, HStack, VStack } from "@chakra-ui/react";
import Image from 'next/image';
import { useFormContext } from 'react-hook-form';

type FormInputProps = {
  name: string;
  label?: string;
  type?: string;
};

const InputBox: React.FC<FormInputProps> = ({
  name,
  label,
  type = "text",
  ...props
}) => {
  const {
    register,
    formState: { errors },
  } = useFormContext();

  return (
    <VStack spacing="1.5" w="100%">
      {label && (<FormLabel htmlFor={name} alignSelf="flex-start">{label}</FormLabel>)}
      <input
        id={name}
        type={type}
        autoFocus
        style={{
          height: '53px',
          borderRadius: '40px',
          paddingLeft: '15px',
          width: '100%',
          backgroundColor: 'white',
          color: 'black',
          outline: 'none',
        }}
        {...register(name)}
        {...props}
      />
      {errors[name] && (
        <FormErrorMessage>
          {errors[name]?.message as string}
        </FormErrorMessage>
      )}
    </VStack>
  );
}

export default InputBox;
