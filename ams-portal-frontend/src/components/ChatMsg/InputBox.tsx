import React from 'react';
import { useState } from 'react';
import { Box, HStack, Toast, VStack, useToast } from '@chakra-ui/react';
import Image from 'next/image';

export default function InputBox({person, loading, inputMessagehandler}: any) {
  const [longDescription, setLongDescription] = useState("");
  const [shortDescription, setShortDescription] = useState("");
  const [updated, setUpdated] = useState(longDescription);
  // console.log('printing from component hello' + person)
  const toast = useToast();
  
  const handleChange = (event: { target: { value: React.SetStateAction<string>; }; }) => {
    setLongDescription(event.target.value);
  };

  const handleChangeShort = (event: { target: { value: React.SetStateAction<string>; }; }) => {
    setShortDescription(event.target.value);
  };

  return (
    <Box
      boxShadow="md"
      position="fixed"
      bottom={5}
      p={1}
      w="80%"
      borderRadius={40}
    >
      <VStack rowGap={5}>
      <input
          id="shortDesc"
          autoFocus
          onChange={handleChangeShort}
          disabled={loading}
          type='text'
          value={shortDescription}
          placeholder="Enter short description/subject/title/topic here... (optional)"
          style={{
            height: '53px',
            borderRadius: '40px',
            paddingLeft: '15px',
            paddingBottom: '10px',
            width: '100%',
            backgroundColor: 'white',
            color: 'grey'
          }}
        />
      <HStack w="100%" justifyContent="center">
        <textarea
          id="longDesc"
          autoFocus
          onChange={handleChange}
          disabled={loading}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && (longDescription != null && longDescription.trim() != "")) {
              inputMessagehandler(shortDescription, longDescription);
              setLongDescription("");
              setShortDescription("");
            } else if (e.key === 'Enter' && !e.shiftKey && (longDescription == null || longDescription.trim() == "")) {
              toast({
                title: "Error",
                description: "Please enter a valid long description.",
                status: "error",
                variant: "left-accent",
                duration: 1000
              });
            }
          }}
          value={longDescription}
          placeholder="Enter long description/body/content here... (required)"
          style={{
            height: '53px',
            borderRadius: '40px',
            paddingLeft: '15px',
            width: '100%',
            backgroundColor: 'white',
            color: 'black',
            outline: 'none',
          }}
        />
        <Image
          src="/send-btn.svg"
          style={{
            cursor: 'pointer',
          }}
          alt="send"
          width={45}
          height={45}
          priority
          onClick={() => {
            if (longDescription != null && longDescription.trim() != "") {
              inputMessagehandler(shortDescription, longDescription);
              setLongDescription("");
              setShortDescription("");
            } else {
              toast({
                title: "Error",
                description: "Please enter a valid long description.",
                status: "error",
                variant: "left-accent",
                duration: 1000
              });
            }
          }}
        />
      </HStack>
      </VStack>
    </Box>
  );
}
