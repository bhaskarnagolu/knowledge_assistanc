import React, { useState } from 'react';
import { HStack, Button, useToast, VStack } from "@chakra-ui/react";
import Image from 'next/image';
import { Feedback } from '@/components';
import { apiRegisterPositiveFeedback } from '@/lib/api-requests';

interface IFeedbackBtnsProps {
  combinedModelId: string;
  accountId: string;
}

export default function FeedbackBtns({
  combinedModelId,
  accountId
}: IFeedbackBtnsProps) {

  const [feedbackStatus, setFeedbackStatus] = useState({
    thumbsUp: false,
    thumbsDown: false,
  });

  const toast = useToast();

  const [openFeedback, onOpenFeedbackModal] = useState(false);

  async function registerPositiveFeedback() {

    setFeedbackStatus({
      thumbsUp: true,
      thumbsDown: false,
    });

    console.log(combinedModelId);

    const result = await apiRegisterPositiveFeedback(accountId, combinedModelId);

    if (result.success) {
      toast({
        title: `Feedback recorded`,
        status: "success",
        variant: "left-accent",
        duration: 2000
      });
    } else {
      toast({
        title: `Could not record feedback. Please try again.`,
        status: "error",
        variant: "left-accent",
        duration: 2000
      });
    }
  }

  // async function registerNegativeFeedback() {
  //   setFeedbackStatus({
  //     thumbsUp: false,
  //     thumbsDown: true,
  //   });
  // }

  return (
    <VStack>
      <HStack alignSelf="flex-end" pt={5}>
        <Button
          type="button"
          variant={feedbackStatus.thumbsUp ? "solid" : "outline"} // Use "solid" when thumbs up is selected
          borderRadius={100}
          onClick={registerPositiveFeedback}
        >
          <Image
            src={ feedbackStatus.thumbsUp ? '/thumbs-up-clicked.svg' : '/thumbs-up.svg' }
            alt="thumbs-up"
            width={15}
            height={10}
            priority
          />
        </Button>
        <Button
          type="button"
          variant={feedbackStatus.thumbsDown ? "solid" : "outline"} // Use "solid" when thumbs down is selected
          borderRadius={100}
          // onClick={() => onOpenFeedbackModal(true)}
          onClick={() => onOpenFeedbackModal(true)}
        >
          <Image
            src={ feedbackStatus.thumbsDown ? '/thumbs-down-clicked.svg' : '/thumbs-down.svg' }
            alt="thumbs-down"
            width={15}
            height={10}
            priority
          />
        </Button>
      </HStack>
      {/** feedback popup */}
      <Feedback
        isFeedbackModalOpen={openFeedback}
        onOpenFeedbackModal={onOpenFeedbackModal}
        combinedModelId={combinedModelId}
        accountId={accountId}
        setNegativeFeedbackStatus={setFeedbackStatus}
      />
  </VStack>
  );
}
