import { useEffect } from "react";
import { apiGetAuthUser } from "./api-requests";
import useStore from "@/store";

export default function useSession() {
  const store = useStore();

  async function fetchUser() {
    try {
      console.log("Fetching User...");
      const user = await apiGetAuthUser();
      store.setAuthUser(user);
      //console.log("User in Drawer Items: ", user);
    } catch (error: any) {
      console.log("Error message:", error.message);
    }
  }

  useEffect(() => {
    console.log("useSession: ", store.authUser)
    if (!store.authUser) {
      fetchUser();
    } else {
      console.log("Existing User: ", store.authUser);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return store.authUser;
}