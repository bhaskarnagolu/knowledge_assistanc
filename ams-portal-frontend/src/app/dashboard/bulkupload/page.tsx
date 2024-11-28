'use client';

import { VStack, Heading, Tab, TabList, TabPanel, TabPanels, Tabs, TabIndicator } from '@chakra-ui/react';
import BulkUploadForm from './bulk-upload-form';
import { useState } from 'react';

export default function BulkUpload() {
  const [tabIndex, setTabIndex] = useState(0);

  return (
    <VStack spacing={5} alignItems="flex-start" flex={1} w="100%">
      <Heading as="h1" color="black" fontWeight="bold" fontSize="4xl">Bulk Upload Questions/Feedback</Heading>
      <Tabs isLazy size='lg' align='start' variant='enclosed' onChange={(index) => setTabIndex(index)}>
        <TabList>
          <Tab _selected={{ color: 'white', bg: 'blue.500' }}>Questions</Tab>
          <Tab _selected={{ color: 'white', bg: 'blue.500' }}>Feedback</Tab>
        </TabList>
        <TabIndicator
          mt="-1.5px"
          height="2px"
          bg="red.500"
          borderRadius="1px"
        />
        <TabPanels>
          <TabPanel>
            <BulkUploadForm key={tabIndex} loadType={"questions"}/>
          </TabPanel>
          <TabPanel>
            <BulkUploadForm key={tabIndex} loadType={"feedback"}/>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </VStack>
  );
}
  