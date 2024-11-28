'use client';

import { useEffect, useState } from "react";
import { ChatMsg, Feedback } from "@/components";
import { VStack, useToast } from "@chakra-ui/react";
import { FilteredUser, modelOutputResponse, predictSingleResponse } from "@/lib/types";
import useSession from "@/lib/useSession";
import { handleApiError } from "@/lib/helpers";
import useStore from "@/store";
import { apiGetPredictSingleOutput, apiPredictSingle } from "@/lib/api-requests";

export default function Chat() {
  const [openFeedback, onOpenFeedback] = useState<boolean>(false);
  const [predictionData, setPredectionData] = useState<predictSingleResponse>();
  const [question, setQuestion] = useState<string>();
  const [title, setTitle] = useState<string>();
  const user: FilteredUser | null  = useSession();
  const toast = useToast();
  const store = useStore();
  let combinedModelId = "";
  const [loading, setLoading] = useState<boolean>(false);

  // do operation every time state of responseData changes
  useEffect(() => {
    if(question) {
      submitQuestion(user?.accountId, question, title || "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
   },[question])


  // function to handle on-click event of chat input box. 
  // it invokes the backend api to submit a question/ticket description
  function handleMessageSubmit(shortDescription: string, longDescription: string) {
    if(longDescription!=null && longDescription!= question) {
      setQuestion(longDescription);
      setTitle(shortDescription);
    }
  }

  // function to fetch model responses against the question/ticket description
  async function submitQuestion (accountId: string | undefined, longDescription: string, shortDescription: string) {
    try {
      setLoading(true);
      const predictionOutput = await apiPredictSingle(accountId ?? 'adani', longDescription, shortDescription); // Using nullish coalescing with fallback value as ''
      console.log(predictionOutput);
      console.log("Got initial response from API");
      // loop until apiGetPredictSingleOutput returns a valid response with 3 model outputs existing and non-empty
      // call apiGetPredictSingleOutput again if the response is invalid and pass predictionOutput.query_id as ticektId
      await new Promise(r => setTimeout(r, 8000));
      let predictionResult = await apiGetPredictSingleOutput(accountId ?? 'ultratech', predictionOutput.query_id);
      console.log(predictionResult);
      let count = 0;
      while(!predictionResult.output || (predictionResult.output.length < 3 || predictionResult.output[0].model_output === "")) {
        console.log("Waiting for model response...");
        await new Promise(r => setTimeout(r, 4000));
        count += 1;
        if (count > 15) {
          throw new Error("Model response timed out.");
        }
        predictionResult = await apiGetPredictSingleOutput(accountId ?? 'ultratech', predictionOutput.query_id);
      }
      console.log("Got final response from API");
      console.log(predictionResult);
      setPredectionData(predictionResult);
      setQuestion("");
      setTitle("");
      setLoading(false);
    } catch (error: any) {
      if (error instanceof Error) {
        handleApiError(error);
      } else {
        console.log("Backend Error message:", error.cod);
      }
      toast({
        title: "Error",
        description: error.message,
        status: "error",
        variant: "left-accent",
        duration: 2000
      });
    } finally {
      setLoading(false);
      store.setRequestLoading(false);
    }
  };

  function makeQuestionString(question: string | undefined, title: string | undefined): string | undefined {
    if (title && title.trim() != "") {
      return "Subject: " + title + "\n" + "Description: " + question;
    } else if (question && question.trim() != "") {
      return question;
    } else {
      return undefined;
    }
  }

  return (
    <VStack flex={1} w="100%">
      {/** show loading animation when loading status is true */}
      {loading?<div className="loader"></div>:null}
      {/** chat content here */}
      <ChatMsg submitHandler={handleMessageSubmit} data={predictionData} query={makeQuestionString(question, title)} loading={loading}/>
    </VStack>
  )
}