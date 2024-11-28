import { FilteredUser, UserLoginResponse, UserResponse, bulkUploadResponse, checkFilesResponse, predictSingleResponse, registerFeedbackResponse, viewBatchResponse, uploadDataResponse, accountDashboardResponse } from "./types";

//const SERVER_ENDPOINT = process.env.SERVER_ENDPOINT || "https://main.dbuddmc5hbu78.amplifyapp.com/";
const SERVER_ENDPOINT = "http://localhost:3000";

async function handleResponse<T>(response: Response): Promise<T> {
  const contentType = response.headers.get("Content-Type") || "";
  const isJson = contentType.includes("application/json");
  const data = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    console.log("Handle Response reported error")
    if (isJson && data.errors !== null) {
      throw new Error(JSON.stringify(data.errors));
    }
    console.log("errormessage: "+data.message + ", statuscode: "+data.cod)
    throw new Error(data.message || response.statusText);
  }

  return data as T;
}

export async function apiLoginUser(credentials: string): Promise<string> {
  const reqheaders: Record<string, string> = {
    'Cache-Control': 'no-cache',
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Accept': '*/*',
  };

  const response = await fetch(`${SERVER_ENDPOINT}/api/auth/login`, {
    method: "POST",
    credentials: "include",
    headers: reqheaders,
    body: credentials,
  });
  console.log("Response in apiLoginUser: ", response)
  
  return handleResponse<UserLoginResponse>(response).then((data) => data.token);
}

export async function apiLogoutUser(): Promise<void> {
  const response = await fetch(`${SERVER_ENDPOINT}/api/auth/logout`, {
    method: "GET",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
    },
  });

  return handleResponse<void>(response);
}

export async function apiGetAuthUser(token?: string): Promise<FilteredUser> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${SERVER_ENDPOINT}/api/users/me`, {
    method: "GET",
    credentials: "include",
    headers,
  });

  return handleResponse<UserResponse>(response).then((data) => data.data.user);
}

export async function apiUploadFile(data: FormData, accountId: string, loadType: string): Promise<uploadDataResponse> {
  const headers: Record<string, string> = {
    'Cache-Control': 'no-cache',
    'Access-Control-Allow-Origin': '*',
    'Accept': '*/*',
  };

  data.append('accountId', accountId);
  data.append('type', loadType);

  const response = await fetch(`${SERVER_ENDPOINT}/api/v1/upload`, {
    method: "POST",
    body: data,
    headers: headers,
  });

  return handleResponse<uploadDataResponse>(response).then((data) => {
    // console.log("got data back", data);
    return data;
  })
}

export async function apiCheckFileExist(accountId: string, loadType: string | number): Promise<boolean> {
    const headers: Record<string, string> = {
      'Cache-Control': 'no-cache',
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Accept': '*/*',
    };
    
    const response = await fetch(`${SERVER_ENDPOINT}/api/v1/checkFiles?accountId=${accountId}`, {
      method: "GET",
      body: null,
      headers: headers
    });

    return handleResponse<checkFilesResponse>(response).then((data) => {
      // console.log("got file existance status", data);
      if(loadType == "ticketData" && data.ticketData)
        return true;
      else if(loadType == "kbData" && data.kbData)
        return true;
      else
        return false;
    })
}

export async function apiRegisterPositiveFeedback(accountId: string, combinedModelId: string): Promise<registerFeedbackResponse> {
  try {
  
    const headers: Record<string, string> = {
      'Cache-Control': 'no-cache',
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Accept': '*/*',
      'X-Auth-Token': process.env.API_SECRET_TOKEN || ""
    };
  
    const response = await fetch(`${SERVER_ENDPOINT}/api/v1/recordPositiveFeedback`, {
      method: "POST",
      body: JSON.stringify({
        "account": accountId,
        "combinedModelId": combinedModelId,
        "score": 1
      }),
      headers: headers
    });

    return handleResponse<registerFeedbackResponse>(response).then((data) => {
      // console.log("got prediction output", data);
      return data;
    })
  } catch {
    return {
      "success": false
    }
  }
}

export async function apiRegisterNegativeFeedback(accountId: string, combinedModelId: string, types: string[], reason?: string): Promise<registerFeedbackResponse> {
  try {
  
    const headers: Record<string, string> = {
      'Cache-Control': 'no-cache',
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Accept': '*/*',
      'X-Auth-Token': process.env.API_SECRET_TOKEN || ""
    };

    let feedback_body;

    if (reason) {

      feedback_body = {
        "account": accountId,
        "combinedModelId": combinedModelId,
        "score": -1,
        "negative_feedback_types": types,
        "negative_feedback_reason": reason
      }
    } else {
      feedback_body = {
        "account": accountId,
        "combinedModelId": combinedModelId,
        "score": -1,
        "negative_feedback_types": types
      }
    }

    console.log(feedback_body)
  
    const response = await fetch(`${SERVER_ENDPOINT}/api/v1/recordNegativeFeedback`, {
      method: "POST",
      body: JSON.stringify(feedback_body),
      headers: headers
    });

    return handleResponse<registerFeedbackResponse>(response).then((data) => {
      // console.log("got prediction output", data);
      return data;
    })
  } catch {
    return {
      "success": false
    }
  }
}

export async function apiGetPredictSingleOutput(accountId: String, ticketId: String): Promise<predictSingleResponse> {
  const headers: Record<string, string> = {
    'Cache-Control': 'no-cache',
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Accept': '*/*',
    'X-Auth-Token': process.env.API_SECRET_TOKEN || ""
  };
  
  const response = await fetch(`${SERVER_ENDPOINT}/api/v1/getPredictSingleSplitResult`, {
    method: "POST",
    body: JSON.stringify({
      "ticketId": ticketId,
      "account": accountId
    }),
    headers: headers
  });

  return handleResponse<predictSingleResponse>(response).then((data) => {
    // console.log("got prediction output", data);
    return data;
  })
}

export async function apiPredictSingle(accountId: string, longDescription: string, shortDescription: string): Promise<predictSingleResponse> {
  const headers: Record<string, string> = {
    'Cache-Control': 'no-cache',
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Accept': '*/*',
    'X-Auth-Token': process.env.API_SECRET_TOKEN || ""
  };
  
  const response = await fetch(`${SERVER_ENDPOINT}/api/v1/predictSingle`, {
    method: "POST",
    body: JSON.stringify({
      "longDescription": longDescription,
      "shortDescription": shortDescription,
      "account": accountId
    }),
    headers: headers
  });

  return handleResponse<predictSingleResponse>(response).then((data) => {
    // console.log("got prediction output", data);
    return data;
  })
}

export async function apiBulkUploadFile(data: FormData, accountId: string, loadType: string): Promise<bulkUploadResponse> {
  const headers: Record<string, string> = {
    'Cache-Control': 'no-cache',
    'Access-Control-Allow-Origin': '*',
    'Accept': '*/*',
    'X-Auth-Token': process.env.API_SECRET_TOKEN || ""
  };

  console.log("account id: ", accountId);

  data.append('accountId', accountId);

  const response = await fetch(`${SERVER_ENDPOINT}/api/v1/${(loadType == "questions" ? "uploadBatch" : "submitBulkFeedback")}`, {
    method: "POST",
    body: data,
    headers: headers
  });

  return handleResponse<bulkUploadResponse>(response).then((data) => {
    return data;
  })
}

export async function apiViewBatch(accountId: string | undefined): Promise<viewBatchResponse> {
  const headers: Record<string, string> = {
    'Cache-Control': 'no-cache',
    'Content-Type': 'Multipart/form-data',
    'Access-Control-Allow-Origin': '*',
    'Accept': '*/*',
    'X-Auth-Token': process.env.API_SECRET_TOKEN || ""
  };

  const response = await fetch(`${SERVER_ENDPOINT}/api/v1/viewBatch?accountId=${accountId}`, {
    method: "GET",
    body: null,
    headers: headers
  });

  return handleResponse<viewBatchResponse>(response).then((data) => {
    // console.log("got data back", data);
    return data;
  })
}

const downloadCSV = (result:any, selectedJob: string) => {
  //console.log("Download code triggered")
  const element = document.createElement("a");
  //console.log(result);
  const file = new Blob([result], {type: 'text/csv'});
  element.href = URL.createObjectURL(file);
  element.download = `file_jobid_${selectedJob}.csv`;
  document.body.appendChild(element); // Required for this to work in FireFox
  element.click();
}

export async function apiDownloadFile(accountId: string | undefined, jobId: string): Promise<any> {
  const headers: Record<string, string> = {
    'Cache-Control': 'no-cache',
    'Content-Type': 'application/octet-stream',
    'Access-Control-Allow-Origin': '*',
    'Accept': '*/*',
    'X-Auth-Token': process.env.API_SECRET_TOKEN || ""
  };

  console.log("jobid:", jobId);

  const response = await fetch(`${SERVER_ENDPOINT}/api/v1/downloadFile?accountId=${accountId}&jobId=${jobId}`, {
    method: "OPTIONS",
    body: null,
    headers: headers
  });

  //console.log(response);

  downloadCSV(await response.blob(), jobId.toString());

  /*return handleResponse<any>(response).then((data) => {
    //console.log("got data back", data);
    downloadCSV(data.body, jobId.toString());
    return data;
  })*/
}

export async function apiAccountDashboard(accountId: string | undefined): Promise<accountDashboardResponse> {
  const headers: Record<string, string> = {
    'Cache-Control': 'no-cache',
    'Content-Type': 'Multipart/form-data',
    'Access-Control-Allow-Origin': '*',
    'Accept': '*/*',
    'X-Auth-Token': process.env.API_SECRET_TOKEN || ""
  };

  const response = await fetch(`http://localhost:3001/api/v1/accountDashboard?accountId=${accountId}`, {
    method: "GET",
    body: null,
    headers: headers
  });

  return handleResponse<accountDashboardResponse>(response).then((data) => {
    return data;
  })
}
