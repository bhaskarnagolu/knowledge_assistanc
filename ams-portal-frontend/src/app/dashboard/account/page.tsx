'use client';

import DynamicTable from '@/components/DynamicTable';
import { apiAccountDashboard } from '@/lib/api-requests';
import { handleApiError } from '@/lib/helpers';
import { FilteredUser, TableData, accountDashboardResponse } from '@/lib/types';
import useSession from '@/lib/useSession';
import useStore from '@/store';
import { VStack, Heading, useToast, Card, CardBody, CardFooter, CardHeader } from '@chakra-ui/react';
import { useEffect, useState } from 'react';

export default function Account() {
  const user: FilteredUser | null  = useSession();
  const toast = useToast();
  const store = useStore();

  const [dataList, setDataList] = useState<TableData[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  //This function will be called on pageLoad event 
  useEffect(() => {
    fetchAccountStatistics();
  }, [user]);

  // function to fetch all the account dashboard statistics from the backend API
  async function fetchAccountStatistics () {
    store.setRequestLoading(true);
    try {
      setLoading(true);
      const accountDashboardResponse: accountDashboardResponse = await apiAccountDashboard(user?.accountId); // Using nullish coalescing with fallback value as ''
      setDataList(accountDashboardResponse.dataList);
    } catch (error: any) {
      if (error instanceof Error) {
        handleApiError(error);
      } else {
        console.log("Error message:", error.cod);
      }
      toast({
        title: "WatsonX.AI API returned timeout or other error while processing. Please try again in a few minutes.",
        description: error.message,
        status: "error",
        variant: "left-accent",
        duration: 2000
      });
    } finally {
      store.setRequestLoading(false);
      setLoading(false);
    }
  };

  return (
      <VStack py={1}>
        {/** show loading animation when loading status is true */}
        {loading?<div className="loader"></div>:null}
        <Heading as="h1" color="black" fontWeight="bold" fontSize="4xl">Account Dashboard</Heading>
          {/** dynamic table content here */}
            <DynamicTable dataList={dataList} loading={loading}/>
      </VStack>
    );
  }
  