import { accountDashboardResponse } from "@/lib/types";
import { Card, CardBody, CardFooter, Center, Spinner, Table, Tbody, Td, Th, Thead, Tr, VStack } from "@chakra-ui/react";
import Image from 'next/image';

export default function DynamicTable (props: accountDashboardResponse) {
  return (
    <>
      <VStack py={1}>
        <Table size="md" variant="solid" colorScheme="teal">
          <Thead>
            <Tr>
              <Th backgroundColor="lightgray">Account</Th>
              <Th backgroundColor="lightgray">Ticket<br/>Data</Th>
              <Th backgroundColor="lightgray">Knowledge<br/>Data</Th>
              <Th backgroundColor="lightgray">Batch Channel<br/>Questions</Th>
              <Th backgroundColor="lightgray">Feedback</Th>
              <Th backgroundColor="lightgray">Performance<br/>Assessment</Th>
            </Tr>
          </Thead>
          <Tbody>
            {props.dataList.map((item, index) => (
              <Tr key={item.id} className={item.tktData === 'yes' ? 'highlight' : ''}>
                <Td textAlign="left">{item.accountName}</Td>
                <Td textAlign="center" className={item.tktData === 'yes' ? 'highlight' : ''}>
                  <Center><Image
                    src={ item.tktData === 'yes' ? '/green-tick.svg' : '/grey-dash.svg' }
                    alt="green-tick | grey-dash"
                    width={20}
                    height={15}
                    priority
                  /></Center>
                </Td>
                <Td textAlign="center" className={item.kbData === 'yes' ? 'highlight' : ''}>
                  <Center>
                    <Image
                      src={ item.kbData === 'yes' ? '/green-tick.svg' : '/grey-dash.svg' }
                      alt="green-tick | grey-dash"
                      width={20}
                      height={15}
                      priority
                    />
                  </Center>
                </Td>
                <Td textAlign="center" className={item.bulkQuestions === 'yes' ? 'highlight' : ''}>
                  <Center>
                    <Image
                      src={ item.bulkQuestions === 'yes' ? '/green-tick.svg' : '/grey-dash.svg' }
                      alt="green-tick | grey-dash"
                      width={20}
                      height={15}
                      priority
                    />
                  </Center>
                </Td>
                <Td textAlign="center" className={item.bulkFeedback === 'yes' ? 'highlight' : ''}>
                  <Center>
                    <Image
                      src={ item.bulkFeedback === 'yes' ? '/green-tick.svg' : '/grey-dash.svg' }
                      alt="green-tick | grey-dash"
                      width={20}
                      height={15}
                      priority
                    />
                  </Center>
                </Td>
                <Td textAlign="center" className={item.assessment === 'yes' ? 'highlight' : ''}>
                  <Center>
                    <Image
                        src={ item.assessment === 'yes' ? '/green-tick.svg' : '/grey-dash.svg' }
                        alt="green-tick | grey-dash"
                        width={20}
                        height={15}
                        priority
                      />
                  </Center>
                </Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
        {/** loading indicator when loading = true */}
        {props.loading && <Spinner size="xl" />}
      </VStack>
    </>
  )
}
