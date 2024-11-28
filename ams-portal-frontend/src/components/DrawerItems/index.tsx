import React from 'react';
import { VStack, HStack, Text } from "@chakra-ui/react";
import Image from 'next/image';
import { FilteredUser } from "@/lib/types";
import { Link } from "@chakra-ui/next-js";

interface IDrawerItemsProps {
  onClose: React.Dispatch<React.SetStateAction<boolean>>;
  user?: FilteredUser | null;
  handleLogout: () => void;
}
export default function DrawerItems({ onClose, user, handleLogout }: IDrawerItemsProps) {
  return (
    <>
      <VStack spacing={5} alignItems="flex-start" flex={1} h="87%">
        <Image
          onClick={() => onClose(false)}
          src="/close.svg"
          alt="close"
          width={25}
          height={25}
          priority
        />
        <HStack>
          <Image
            src="/portfolio.svg"
            alt="Dashboard"
            width={25}
            height={25}
            priority
          />
          <Text color="black" fontWeight="bold" fontSize={16}>
            <Link href="/dashboard" color="black" fontWeight="bold" fontSize={16} >Dashboard</Link>
          </Text>
        </HStack>
        <HStack>
          <Image
            src="/add-comment.svg"
            alt="new-chat"
            width={25}
            height={25}
            priority
          />
          <Text color="black" fontWeight="bold" fontSize={16}>
            {user ? (
                <Link href="/dashboard/chat" color="black" fontWeight="bold" fontSize={16} >New Ticket</Link>
              ) : (
                // null /*<Link href="/login" color="black" fontWeight="bold" fontSize={16}>New Chat</Link>*/
                <Link href="/login" color="black" fontWeight="bold" fontSize={16}>New Ticket</Link>
            )}
          </Text>
        </HStack>
        <HStack>
          <Image
            src="/recently-viewed.svg"
            alt="saved-chat"
            width={25}
            height={25}
            priority
          />
          <Text color="black" fontWeight="bold" fontSize={16}>
            Previous Tickets
          </Text>
        </HStack>
        <HStack>
          <Image
            src="/add-comment.svg"
            alt="upload-single"
            width={25}
            height={25}
            priority
          />
          <Text color="black" fontWeight="bold" fontSize={16}>
            {user ? (
                <Link href="/dashboard/upload" color="black" fontWeight="bold" fontSize={16} >Upload Ticket/KB Data</Link>
              ) : (
                <Link href="/login" color="black" fontWeight="bold" fontSize={16}>Upload Ticket/KB Data</Link>
            )}
          </Text>
        </HStack>
        <HStack>
          <Image
            src="/add-comment.svg"
            alt="upload-bulk"
            width={25}
            height={25}
            priority
          />
          <Text color="black" fontWeight="bold" fontSize={16}>
            {user ? (
                <Link href="/dashboard/bulkupload" color="black" fontWeight="bold" fontSize={16}>Bulk Upload Questions/Feedback</Link>
              ) : (
                <Link href="/login" color="black" fontWeight="bold" fontSize={16}>Bulk Upload Questions/Feedback</Link>
            )}
          </Text>
        </HStack>
        {/**<HStack>
          <Image
            src="/add-comment.svg"
            alt="account-dashboard"
            width={25}
            height={25}
            priority
          />
          <Text color="black" fontWeight="bold" fontSize={16}>
            {user ? (
                <Link href="/dashboard/account" color="black" fontWeight="bold" fontSize={16}>Account Dashboard</Link>
              ) : (
                <Link href="/login" color="black" fontWeight="bold" fontSize={16}>Account Dashboard</Link>
            )}
          </Text>
              </HStack>**/}
      </VStack>
      <VStack alignItems="flex-start">
        <HStack>
          {user && (
            <VStack alignItems="flex-start">
              <Text color="black" fontWeight="semibold" fontSize={12} pl={3}>
                {user.name}
              </Text>
              <Text color="black" fontSize={10} pl={3}>
                {user.role}
              </Text>
            </VStack>
          )}
        </HStack>
        <HStack borderTopWidth={1} w="100%" pt={3}>
          <Image
            src="/logout.svg"
            alt="logout"
            width={25}
            height={25}
            priority
          />
          {user ? (
            <Link href="#" color="black" fontWeight="bold" fontSize={16} onClick={(e) => { e.preventDefault(); handleLogout() }}>Logout</Link>
          ) : (
            <Link href="/login" color="black" fontWeight="bold" fontSize={16}>Login</Link>
          )}
        </HStack>
      </VStack>
    </>
  );
}
