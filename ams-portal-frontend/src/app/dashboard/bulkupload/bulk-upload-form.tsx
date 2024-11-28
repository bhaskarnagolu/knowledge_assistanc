import { Key, useEffect, useRef, useState, RefObject } from "react";
import { useForm, SubmitHandler, FormProvider } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Button, VStack, Box, useToast, HStack, Card, Stack } from "@chakra-ui/react";
import { UploadFormInput, UploadFormSchema, ACCEPTED_INPUT_FILES, FILE_KEY } from "@/lib/validations/upload.schema";
import FileUpload from "@/components/FileUpload";
import { apiBulkUploadFile, apiUploadFile } from "@/lib/api-requests";
import useStore from "@/store";
import { handleApiError } from "@/lib/helpers";
import { Table, Tbody, Td, Th, Thead, Tr } from '@chakra-ui/react';
import React from "react";
import useSession from "@/lib/useSession";
import { FilteredUser, bulkUploadResponse } from "@/lib/types";

export default function BulkUploadForm({loadType}: any) {
  const toast = useToast();

  const store = useStore();

  const [files, setFiles] = useState<File[]>([]);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const formRef: RefObject<HTMLFormElement> = React.useRef(null);

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
      setSelectedFile(files[0])
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  },[files])


  // Function to Upload file using backend api
  async function UploadFormFunction(formData: UploadFormInput) {
    store.setRequestLoading(true);
    try {
      const data = new FormData()
      data.set(FILE_KEY, formData[FILE_KEY])
      console.log("Data: ", data)

      //const bulkUploadOutput:bulkUploadResponse = await apiBulkUploadFile(data, user?.accountId || "", loadType);
      const bulkUploadOutput: bulkUploadResponse = await apiBulkUploadFile(data, user?.accountId || "", loadType);
      //(bulkUploadOutput.fail_reason=="")? `Data ${bulkUploadOutput.status} for batch_id: ${bulkUploadOutput.batch_id}`: `Upload failed with reason: ${bulkUploadOutput.fail_reason}`
      toast({
        title: (bulkUploadOutput.fail_reason=="")? `Data uploaded successfully`: `Upload failed with reason: ${bulkUploadOutput.fail_reason}`,
        status: (bulkUploadOutput.fail_reason=="") ? "success" : "error",
        variant: "left-accent",
        duration: 2000
      });
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
    UploadFormFunction(values);
  }

  return (
    <Box>
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
                            <Button type="submit" size="sm" color="teal" variant="striped">
                              {'Add'}
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
    </Box>
  )
}