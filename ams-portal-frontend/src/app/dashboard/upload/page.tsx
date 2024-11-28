'use client';

import { VStack, Heading } from '@chakra-ui/react';
import UploadForm from './upload-form';

export default function Upload() {
    return (
      <VStack spacing={5} alignItems="flex-start" flex={1}>
        <Heading as="h1" color="black" fontWeight="bold" fontSize="4xl">Upload Ticket/KB Data</Heading>
        <UploadForm />
      </VStack>
    );
  }
  