import React, { useState, useRef, useEffect } from 'react';
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalFooter,
  Card,
  Text,
  ModalBody,
  ModalCloseButton,
  Button,
  HStack,
  useToast,
} from "@chakra-ui/react";
import Image from 'next/image';
import Reasons from '@/components/Feedback/reasons';
import { reasons } from '@/lib/utils';
import { IReasons } from '@/models';
import { apiRegisterNegativeFeedback } from '@/lib/api-requests';

interface IFeedbackProps {
  isFeedbackModalOpen: boolean;
  onOpenFeedbackModal: React.Dispatch<React.SetStateAction<boolean>>;
  combinedModelId: string;
  accountId: string;
  setNegativeFeedbackStatus: any;
}
export default function Feedback({
  isFeedbackModalOpen,
  onOpenFeedbackModal,
  combinedModelId,
  accountId,
  setNegativeFeedbackStatus
}: IFeedbackProps) {
  const [feedback, setFeedback] = useState<IReasons[]>(reasons);
  const [textFeedback, setTextFeedback] = useState<string>("");
  
  const toast = useToast();

  async function resetFeedback() {
    const tmpFeedback = reasons;
    for (let i=0; i< tmpFeedback.length; ++i) {
      tmpFeedback[i].isSelected = false;
    }
    setFeedback(tmpFeedback);
  }

  useEffect(() => {
    if (!isFeedbackModalOpen) {
      resetFeedback();
    }
  }, [isFeedbackModalOpen]);

  async function registerNegativeFeedback() {

    try {

      let types = []

      for (let i=0; i < feedback.length; ++i) {
        if (feedback[i].isSelected) types.push(feedback[i].title);
      }

      if ((!textFeedback || !textFeedback.length) && !types.length || (types.includes('Other (please elaborate)') && !textFeedback.length)) {
        toast({
          title: `Please select at least one category or input a description to record feedback.`,
          status: "info",
          variant: "top-accent",
          duration: 3000
        });
        return;
      }

      console.log(combinedModelId);
      console.log(textFeedback);
      console.log(feedback);

      let result;

      if (textFeedback && textFeedback.length)
      {
        result = await apiRegisterNegativeFeedback(accountId, combinedModelId, types, textFeedback);
      }
      else {
        result = await apiRegisterNegativeFeedback(accountId, combinedModelId, types);
      }

      //invoke the parent's callback handler function and pass feedback status to it
      setNegativeFeedbackStatus({
        thumbsUp: false,
        thumbsDown: true,
      });

      toast({
        title: `Feedback recorded.`,
        status: "success",
        variant: "left-accent",
        duration: 2000
      });

      onOpenFeedbackModal(!isFeedbackModalOpen);
    } catch {
      toast({
        title: `Could not record feedback. Please try again.`,
        status: "error",
        variant: "left-accent",
        duration: 2000
      });
    }
  }
  return (
    <Modal
      isCentered
      isOpen={isFeedbackModalOpen}
      onClose={() => onOpenFeedbackModal(!isFeedbackModalOpen)}
      size="3xl"
    >
      <ModalOverlay />
      <ModalContent>
        <HStack w="100%" p={5} borderBottom="1px solid lightgray">
          <Image
            src="/thumbs-down-icon.svg"
            alt="thumbs-down-icon"
            width={30}
            height={25}
            priority
          />
          <Text color="black" fontSize={24} fontWeight="bold">
            Why is the response not useful?
          </Text>
        </HStack>
        <ModalCloseButton color="black" />
        <ModalBody>
          {/**feedback selection */}
          <Reasons reasons={reasons} onHandleReason={setFeedback} />
          {/**feedback textarea field */}
          <textarea
            placeholder="Provide additional feedback..."
            autoFocus
            onChange={event => setTextFeedback(event?.target.value)}
            style={{
              borderRadius: '10px',
              paddingLeft: '10px',
              marginTop: '10px',
              padding: '10px',
              height: '80px',
              width: '100%',
              color: 'black',
              border: '1px solid lightgray',
              outline: 'none',
            }}
          />
        </ModalBody>
        <ModalFooter>
          <Button
            colorScheme="messenger"
            borderRadius={40}
            mr={3}
            fontSize={16}
            onClick={registerNegativeFeedback}
          >
            Submit
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}
