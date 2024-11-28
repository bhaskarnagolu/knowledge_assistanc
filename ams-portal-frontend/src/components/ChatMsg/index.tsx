import React, { useEffect, useState } from 'react';
import { Toast, VStack, useToast } from "@chakra-ui/react";
import InputBox from './InputBox';
import Messages from './Messages';
import { Message, modelOutputResponse, predictSingleResponse } from '@/lib/types';
import { Spinner } from '@chakra-ui/react';

interface IChatMsgProps {
  submitHandler: any;
  data: predictSingleResponse | undefined;
  query: string | undefined;
  loading: boolean;
}
export default function ChatMsg({ submitHandler, data, query, loading }: IChatMsgProps) {

  // const [predictionHistory, setPredictionHistory] = useState<modelOutputResponse[][]>([]);
  // const [queryHistory, setQueryHistory] = useState<string[]>([]);
  const [messageHistory, setMessageHistory] = useState<Message[]>([]);
  const toast = useToast();
  const [openFeedback, onOpenFeedback] = useState<boolean>(false);
  
  // do operation every time state of query changes
  useEffect(()=> {
    if(query) {
      // setQueryHistory(prevQueryHistory => [...prevQueryHistory, query])
      const newMessage: Message = {
        isResponse: false,
        text: query,
        timestamp: new Date().getTime()
      };
      setMessageHistory(prevMessageHistory => [...prevMessageHistory, newMessage]);
    }
  },[query])

  // do operation every time state of data changes
  useEffect(() => {
   if(data && data.output && data.output.length == 3) {
    //  setPredictionHistory(prevPredictionHistory => [...prevPredictionHistory, data.output]);
     data.output.map((message:modelOutputResponse, index) => (
        setMessageHistory(prevMessageHistory => [...prevMessageHistory, {
          isResponse: true,
          heading: message.model_str,
          text: message.model_output,
          timestamp: new Date().getTime(),
          combinedModelId: data?.query_id + "_" + index
        }])
    ));
   } else if (data) {
    toast(
      {
        title: `WatsonX.AI API returned timeout or other error while processing. Please try again in a few minutes.`,
        status: "error",
        variant: "left-accent",
        duration: 3000
      }
    )
   }
   // eslint-disable-next-line react-hooks/exhaustive-deps
  },[data])

  return (
    <VStack pr={52} w="100%">
      <VStack w="100%" pt={20} spacing={5} pb={100}>
      {messageHistory.map((resp: Message, index) => (
          <Messages key={data?.query_id + "_" + index} heading={resp.heading ? resp.heading : undefined} data={resp.text} isAnswer={resp.isResponse} combinedModelId={resp.combinedModelId || ""}></Messages>
        ))
      }
      {/** loading indicator when loading = true */}
      {loading && <Spinner size="xl" />}
    </VStack>
      {/** input box */}
      <InputBox loading={loading} person={{ name: 'Lin Lanying', imageId: '1bX5QH6' }} inputMessagehandler={submitHandler} />
    </VStack>
  );
}