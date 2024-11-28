import React, { useEffect, useState } from 'react';
import { HStack, Card } from "@chakra-ui/react";
import Image from 'next/image';
import { FeedbackBtns } from '@/components';
import useSession from '@/lib/useSession';

interface IMessagesProps {
  data?: string;  // data prop should contain the question
  isAnswer?: boolean;
  combinedModelId: string;
}

const Messages: React.FC<IMessagesProps> = ({
  isAnswer = false,
  data,
  combinedModelId
}) => {
  const user = useSession();
  const [fetchedData, setFetchedData] = useState<string | null>(null);

  // Log session data
  useEffect(() => {
    console.log('User session data:', user);
  }, [user]);

  // Log and set the data prop
  useEffect(() => {
    console.log("Data prop received:", data);
    // Only set the data once, and prevent multiple renders
    if (data && data !== fetchedData) {
      setFetchedData(data);
    }
  }, [data, fetchedData]);

  const renderImage = (src: string, alt: string) => (
    <Image src={src} alt={alt} width={40} height={40} style={{ width: '40px', height: 'auto' }} priority />
  );

  const formatText = (text: string) => {
    const sections = text.split(/(?:Root Cause|Solution|Note):/i);
    const rootCause = sections[1]?.trim() || '';
    const solution = sections[2]?.trim() || '';

    return (
      <div>
        {rootCause && (
          <div>
            <strong>Root Cause:</strong><br />
            {rootCause}<br /><br />
          </div>
        )}
        {solution && (
          <div>
            <strong>Solution:</strong><br />
            {/* Render the solution without repeating */}
            <div style={{ marginLeft: '20px' }}>
              {solution.split(/\n/).map((step, index) => (
                <div key={index}>{step.trim()}</div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <HStack w="100%" justifyContent={isAnswer ? "flex-start" : "flex-end"}>
      {isAnswer && renderImage("/bot.svg", "bot avatar")}

      <Card p={2} borderRadius={10} bg="#ffffff" id={isAnswer ? "answer" : "question"}>
        {/* Display data for answer or question */}
        {fetchedData ? (
          <>
            {isAnswer ? (
              // Format and display fetched answer
              formatText(fetchedData)
            ) : (
              // Display question directly without formatting
              <p>{fetchedData}</p>
            )}
          </>
        ) : (
          <p>No data available to display.</p>
        )}

        {/* Render feedback buttons if it's an answer and user session exists */}
        {isAnswer && user && (
          <FeedbackBtns combinedModelId={combinedModelId} accountId={user.accountId} />
        )}
      </Card>

      {!isAnswer && renderImage("/avatar.svg", "user avatar")}
    </HStack>
  );
};

export default Messages;
