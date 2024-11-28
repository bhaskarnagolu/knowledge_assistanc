import { useEffect } from "react";
import { useForm, SubmitHandler, FormProvider } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { Link } from '@chakra-ui/next-js'
import { Box, useToast, VStack, Flex, Button } from "@chakra-ui/react";
import { LoginUserInput, LoginUserSchema } from "@/lib/validations/user.schema";
import { apiLoginUser } from "@/lib/api-requests";
import useStore from "@/store";
import { handleApiError } from "@/lib/helpers";
import { InputBox } from "@/components";

export default function LoginForm() {
  const store = useStore();
  const router = useRouter();
  const toast = useToast();

  const methods = useForm<LoginUserInput>({
    resolver: zodResolver(LoginUserSchema),
  });

  const {
    reset,
    handleSubmit,
    formState: { isSubmitSuccessful },
  } = methods;

  useEffect(() => {
    store.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function LoginUserFunction(credentials: LoginUserInput) {
    store.setRequestLoading(true);
    try {
      await apiLoginUser(JSON.stringify(credentials));

      toast({
        title: "Logged in successfully",
        status: "success",
        variant: "left-accent"
      });
      router.push("/dashboard");
      return;
    } catch (error: any) {
      console.log(error);
      if (error instanceof Error) {
        handleApiError(error);
      } else {
        console.log("Error message:", error.message);
      }
      toast({
        title: "Error",
        description: error.message + " . Please try navigating to the homepage again.",
        status: "error",
        variant: "left-accent"
      });
    } finally {
      store.setRequestLoading(false);
    }
  }

  const onSubmitHandler: SubmitHandler<LoginUserInput> = (values) => {
    LoginUserFunction(values);
  };

  return (
    <FormProvider {...methods}>
      <form
        onSubmit={handleSubmit(onSubmitHandler)}>
        <VStack spacing="3">
          <InputBox label="Email" name="email" type="email" />
          <InputBox label="Password" name="password" type="password" />
          <Flex justify="flex-end">
            <Link href="#" className="">
              Forgot Password?
            </Link>
          </Flex>
          <Box pt={10}>
            <Button type="submit" size="lg" colorScheme="twitter" isLoading={store.requestLoading} variant="outline">
              Login
            </Button>
          </Box>
        </VStack>
      </form>
    </FormProvider>
  );
}