'use client';

import { VStack, Text, Box, Button, Card, CardBody, CardHeader, HStack, Table, Tbody, Td, Th, Thead, Tr, useToast, Divider, CardFooter, Heading } from "@chakra-ui/react";
import { Link } from "@chakra-ui/next-js";
import { Key, useEffect, useState } from "react";
import { FilteredUser, jobResponse, viewBatchResponse } from "@/lib/types";
import useSession from "@/lib/useSession";
import { apiDownloadFile, apiViewBatch } from "@/lib/api-requests";
import { handleApiError } from "@/lib/helpers";
import useStore from "@/store";

export default function Dashboard() {
  const [files, setFiles] = useState<File[]>([]);
  const user: FilteredUser | null  = useSession();
  const toast = useToast();
  const store = useStore();

  const [data, setData] = useState<viewBatchResponse>();
  const [currentJobs, setCurrentJobs] = useState<jobResponse[]>([])
  const [completedJobs, setCompletedJobs] = useState<jobResponse[]>([])
  const [fileStream, setFileStream] = useState();
  const [selectedJob, setSelectedJob] = useState<string>("");

  //This function will be called on pageLoad event 
  useEffect(() => {
    fetchBatchDetails();
  }, [user]);

  // This will be called when the data has been fetched
  useEffect(() => {
    if (data) {
      // Do something with the data
      setCurrentJobs(data.current_jobs);
      setCompletedJobs(data.completed_jobs);
    }
  }, [data]);

  // This will be called when a new fileStream result is populated/changed
  useEffect(() => {
    if(fileStream) {
      // console.log("trying to download filestream")
      downloadCSV(fileStream);
    }
  }, [fileStream])

  // This will reset the job selection and must be executed before downloading the  
  async function resetSelection(jobId: string) {
    //console.log("resetting selection");
    //console.log("jobId from UI:", jobId);
    setSelectedJob(jobId);
    setFileStream(await apiDownloadFile(user?.accountId, jobId)); // Using nullish coalescing with fallback value as ''
  }

  // function to fetch all the batch job details from the database
  async function fetchBatchDetails () {
    //store.setRequestLoading(true);
    try {
      const batchResponse: viewBatchResponse = await apiViewBatch(user?.accountId); // Using nullish coalescing with fallback value as ''
      setData(batchResponse);
    } catch (error: any) {
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
      //store.setRequestLoading(false);
    }
  };

  // handler function to download the CSV
  async function handleDownload() {
    //store.setRequestLoading(true);
    try {
      // setFileStream(await apiDownloadFile(user?.accountId, selectedJob)); // Using nullish coalescing with fallback value as ''
      toast({
        title: "Success",
        description: "File downloaded successfully",
        status: "success",
        variant: "left-accent",
        duration: 2000
      });
    } catch (error: any) {
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
      //store.setRequestLoading(false);
    }
  }
  
  //when a new result is populated, take the csv string contained in result and trigger a download of a csv file that has it as the content
  const downloadCSV = (result:any) => {
    console.log("Download code triggered")
    const element = document.createElement("a");
    //console.log(result);
    const file = new Blob([result], {type: 'text/csv'});
    element.href = URL.createObjectURL(file);
    element.download = `file_jobid_${selectedJob}.csv`;
    document.body.appendChild(element); // Required for this to work in FireFox
    element.click();
  }
  
  return (
    <VStack py={1}>
      <Text>Welcome to the NextGen AMS Knowledge Assistant!</Text>
      {/* <Link href="/dashboard/upload" color="cyan.700">Check out the upload page</Link>  */}
      <Box w='100%'>
        <div>
          <HStack w="100%" justifyContent="flex-start" flex="space-between">
          {
            data && (
            <Card className="max-w-[400px]" p={2} borderRadius={10} bg="#ffffff" sx={{width: '100%',}}>
              <CardHeader color="warning" className="flex gap-3">
                <Heading size='sm'>Current Jobs</Heading>
              </CardHeader>
              <Divider/>
              <CardBody>
                <Table size="sm" variant="solid" colorScheme="teal">
                  <Thead>
                    <Tr>
                      <Th backgroundColor="gray.50">SN.</Th>
                      <Th backgroundColor="gray.50">Job ID</Th>
                      <Th backgroundColor="gray.50">Status</Th>
                      <Th backgroundColor="gray.50">Error</Th>
                      <Th backgroundColor="gray.50">Download Link</Th>
                    </Tr>
                  </Thead>
                  <Tbody>
                  {
                    currentJobs ? currentJobs.map((job: jobResponse, key: Key | any) => {
                      return (
                      <Tr key={key} >
                        <Td>{++key}</Td>
                        <Td>{job.batch_id}</Td>
                        <Td>{job.status}</Td>
                        <Td>{job.error}</Td>
                        <Td>
                          {job.status == "ready"?<Link textDecor="none" color="blue.500" _hover={{textDecoration: "underline", fontWeight: "bold", color: "red.700"}} onClick={async ()=> {await resetSelection(job.batch_id); handleDownload()}} href={""}>Click here</Link>:""}
                        </Td>
                      </Tr>
                      )
                    }) : null
                  }
                  </Tbody>
                </Table>
              </CardBody>
              <Divider />
              <CardFooter height={55} m={[2, -3]}>
                <small>
                  Total Current jobs: {currentJobs ? currentJobs.length : 0}
                </small>
              </CardFooter>
            </Card>
            )
          }
          </HStack>
        </div>
      </Box>
      <Box w='100%' >
        <div>
          <HStack w="100%" justifyContent="flex-start" flex="space-between">
          {
            data && (
            <Card className="max-w-[400px]" p={2} borderRadius={10} bg="#ffffff" sx={{width: '100%',}}>
              <CardHeader color="warning" className="flex gap-3">
                <Heading size='sm'>Completed Jobs</Heading>
              </CardHeader>
              <Divider/>
              <CardBody>
                <Table size="sm" variant="solid" colorScheme="teal">
                  <Thead>
                    <Tr>
                    <Th backgroundColor="gray.50">SN.</Th>
                      <Th backgroundColor="gray.50">Job ID</Th>
                      <Th backgroundColor="gray.50">Status</Th>
                      <Th backgroundColor="gray.50">Error</Th>
                      <Th backgroundColor="gray.50">Download Link</Th>
                    </Tr>
                  </Thead>
                  <Tbody>
                  {
                    completedJobs ? completedJobs.map((job: jobResponse, key: Key | any) => {
                      return (
                      <Tr key={key} >
                        <Td>{++key}</Td>
                        <Td>{job.batch_id}</Td>
                        <Td>{job.status}</Td>
                        <Td>{job.error}</Td>
                        <Td>
                          {job.status == "ready"?<Link textDecor="none" color="blue.500" _hover={{textDecoration: "underline", fontWeight: "bold", color: "red.700"}} onClick={async () => {await resetSelection(job.batch_id); handleDownload()}} href={""}>Click here</Link>:""}
                        </Td>
                      </Tr>
                      )
                    }) : null
                  }
                  </Tbody>
                </Table>
              </CardBody>
              <Divider/>
              <CardFooter height={55} m={[2, -3]}>
                <small>
                  Total Completed jobs: {completedJobs ? completedJobs.length : 0}
                </small>
              </CardFooter>
            </Card>
            )
          }
          </HStack>
        </div>
      </Box>
    </VStack>
  );
}
