import { Key, useEffect, useRef, useState, RefObject } from "react";
import { useForm, SubmitHandler, FormProvider } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Button, VStack, Box, useToast, HStack, Card, Stack } from "@chakra-ui/react";
import { UploadFormInput, UploadFormSchema, ACCEPTED_INPUT_FILES, FILE_KEY } from "@/lib/validations/upload.schema";
import FileUpload from "@/components/FileUpload";
import { apiCheckFileExist, apiGetAuthUser, apiUploadFile } from "@/lib/api-requests";
import useStore from "@/store";
import { handleApiError } from "@/lib/helpers";
import { Table, Tbody, Td, Th, Thead, Tr } from '@chakra-ui/react';
import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogContent,
  AlertDialogOverlay,
  useDisclosure,
} from '@chakra-ui/react';
import React from "react";
import useSession from "@/lib/useSession";
import { FilteredUser } from "@/lib/types";

import CustomRadio from "@/components/CustomRadio";
import { useRadioGroup, Text } from "@chakra-ui/react"

export default function UploadForm() {
  const [files, setFiles] = useState<File[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileExistsInDatabase, setFileExistsInDatabase] = useState(false);

  const [consentApproved, setConsentApproved] = useState(false);

  const { isOpen, onOpen, onClose } = useDisclosure();
  const cancelRef = React.useRef<HTMLButtonElement | null>(null);
  const formRef: RefObject<HTMLFormElement> = React.useRef(null);

  const toast = useToast();
  const store = useStore();

  const user: FilteredUser | null  = useSession();

  const methods = useForm<UploadFormInput>({
    resolver: zodResolver(UploadFormSchema),
  });

  const {
    reset,
    handleSubmit,
    formState: { isSubmitSuccessful },
  } = methods;

  /*useEffect(() => {
    if (isSubmitSuccessful) {
      reset();
    }
    store.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isSubmitSuccessful]);*/

  // do operation every time state of files changes
  useEffect(() => {
    if(files.length>0) {
      handleOnChangeEvent();
      setSelectedFile(files[0])
      setConsentApproved(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  },[files])

  // do operation every time consent for the file upload is marked approved
  useEffect(() => {
    if(consentApproved) {
      const obj: UploadFormInput = {[FILE_KEY]: files[0]};
      UploadFormFunction(obj);
      setConsentApproved(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [consentApproved])

  // Function to Upload file using backend api
  async function UploadFormFunction(formData: UploadFormInput) {
    store.setRequestLoading(true);
    try {
      const data = new FormData()
      data.set(FILE_KEY, formData[FILE_KEY])
      console.log("Data: ", data)
      
      const resp = await apiUploadFile(data, user?.accountId || "", value.toString());
      
      if (resp.status == "failed") {
        toast({
          title: "Error",
          description: resp.fail_reason,
          status: "error",
          variant: "left-accent",
          duration: 4000
        });  
      } else {
        toast({
          title: "Data uploaded successfully",
          description: resp.status,
          status: "success",
          variant: "left-accent",
          duration: 2000
        });
      }
    } catch (error: any) {
      console.log(error);
      if (error instanceof Error) {
        handleApiError(error);
      } else {
        console.log("Error message:", error.message);
      }
      toast({
        title: "Error",
        description: error.message,
        status: "error",
        variant: "left-accent",
        duration: 2000
      });
    } finally {
      store.setRequestLoading(false);
    }
  }

  // This form submit handler comes into effect when the selected file does not exist in database
  const onSubmitHandler: SubmitHandler<UploadFormInput> = (values) => {
    if(!fileExistsInDatabase) {
      UploadFormFunction(values);
    }
  }

  // function to handle on-change event of File input. 
  // it invokes the backend api to check if selected file does exist or not
  const handleOnChangeEvent = () => {
    if(files.length>0) {
      checkFileInDatabase();
    }
  }

  // function to check if the file exists in the database
  async function checkFileInDatabase () {
    store.setRequestLoading(true);
    try {
      const isFileInDatabase = await apiCheckFileExist(user?.accountId || "", value.toString()); // Using nullish coalescing with fallback value as ''
      setFileExistsInDatabase(isFileInDatabase);
    } catch (error: any) {
      console.log(error);
      if (error instanceof Error) {
        handleApiError(error);
      } else {
        console.log("Error message:", error.cod);
      }
      toast({
        title: "Error",
        description: error.message,
        status: "error",
        variant: "left-accent",
        duration: 2000
      });
    } finally {
      store.setRequestLoading(false);
    }
  };


  // Handle submission of the new file to be added to the database
  const handleSubmitAdd = async () => {
    // console.log("add action called")
  };

  // Handle submission to replace/override the existing file in the database
  const handleSubmitReplaceOverride = async () => {
    setConsentApproved(false);
    onOpen(); // Open the AlertDialog for user consent
  };

  const handleAlertDialogSubmit = async (event: React.BaseSyntheticEvent) => {
    event.preventDefault();
    if (fileExistsInDatabase) {
      // Handle API call for replacing/overriding the file
      setConsentApproved(true);
    }
    onClose(); // Close the AlertDialog after submission
  };

  const removeFile = (i: any) => {
    setFiles(files.filter((x: { name: any; }) => x.name !== i));
  }


  // Radio group related properties/methods
  const radioOptions = [
    { name: 'ticketData', alttext: 'Ticket Data', image: 'https://cdn-icons-png.flaticon.com/512/4196/4196323.png' },
    { name: 'kbData', alttext: 'KB Data', image: 'https://cdn-icons-png.flaticon.com/512/2600/2600314.png' },
  ]

  const handleChange = (value: any) => {
    toast({
      title: `Upload category got changed to ${value}!`,
      status: 'success',
      duration: 2000,
    })
  }

  const { value, getRadioProps, getRootProps } = useRadioGroup({
    defaultValue: 'ticketData',
    onChange: handleChange,
  })


  return (
    <Box>
      <Stack {...getRootProps()}>
        <HStack>
          {radioOptions.map((avatar) => {
              return (
                  <CustomRadio
                      key={avatar.name}
                      image={avatar.image}
                      alttext={avatar.alttext}
                      {...getRadioProps({ value: avatar.name })} />
              )
          })}
        </HStack>
        <Text></Text><Text></Text><Text></Text>
      </Stack>
      <FormProvider {...methods}>
        <form ref={formRef} onSubmit={handleSubmit(onSubmitHandler)}>
          <VStack spacing="5">
            <FileUpload files={files} setFiles={setFiles} name={FILE_KEY} accept={ACCEPTED_INPUT_FILES.toString()} multiple={false} />
            <div className="flex flex-wrap gap-2 mt-2">
            {
              files.length>0 && (
                <Box><div>
                  <HStack w="100%" justifyContent="flex-end">
                  <Card p={2} borderRadius={10} bg="#ffffff">
                  <Table size="sm" variant="solid" colorScheme="teal">
                    <Thead>
                      <Tr>
                        <Th>SN.</Th>
                        <Th>File Name</Th>
                        <Th>Action</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                    {
                      files.map((file: File, key: Key | any) => {
                        return (
                        <Tr key={key} >
                          <Td>{++key}</Td>
                          <Td>{file.name}</Td>
                          <Td>
                            <Button type="submit" size="sm" color="teal" variant="striped" onClick={fileExistsInDatabase ? handleSubmitReplaceOverride: handleSubmitAdd}>
                              {fileExistsInDatabase ? 'Upload/Override' : 'Add'}
                            </Button>
                          </Td>
                        </Tr>
                        )
                      })
                    }
                    </Tbody>
                  </Table>
                  </Card>
                  </HStack>
                </div></Box>
              )
            }
            </div>
          </VStack>
        </form>
      </FormProvider>
      <AlertDialog isOpen={isOpen} leastDestructiveRef={cancelRef} onClose={onClose} motionPreset='slideInBottom' isCentered >
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader>Confirm Submission</AlertDialogHeader>
            <AlertDialogBody>
              {fileExistsInDatabase
                ? 'Are you sure you want to replace/override the existing file?'
                : 'Are you sure you want to add the new file?'}
            </AlertDialogBody>
            <AlertDialogFooter>
              <Button ref={cancelRef} onClick={onClose}>Cancel</Button>
              <Button colorScheme="red" onClick={handleAlertDialogSubmit} ml={3}>
                Confirm
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </Box>
  )
}