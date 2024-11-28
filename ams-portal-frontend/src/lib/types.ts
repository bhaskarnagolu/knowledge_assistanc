export interface FilteredUser {
  id: string;
  name: string;
  email: string;
  accountId: string;
  role: string;
  verified: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface UserResponse {
  status: string;
  data: {
    user: FilteredUser;
  };
}

export interface UserLoginResponse {
  status: string;
  token: string;
}

export interface checkFilesResponse {
  ticketData: string;
  kbData: string;
}

export interface predictSingleResponse {
  query_id: String;
  output: modelOutputResponse[];
}

export interface registerFeedbackResponse {
  success: boolean;
}

export interface uploadDataResponse {
  account_id: string;
  fail_reason?: string;
  status: "failed" | "success";
  type: "ticketData" | "kbData";
}

export interface modelOutputResponse {
  model_str: string;
  model_output: string;
}

export interface bulkUploadResponse {
  batch_id: number;
  status: string;
  fail_reason: string
}

export interface Message {
  isResponse: boolean;
  heading?: string | undefined;
  text: string;
  timestamp: any;
  combinedModelId?: string;
}

export interface viewBatchResponse {
  current_jobs: jobResponse[];
  completed_jobs: jobResponse[];
}

export interface jobResponse {
  batch_id: string;
  status: string;
  error: string;
}

export interface TableData {
  id: number;
  accountName: string;
  tktData: string;
  kbData: string;
  bulkQuestions: string;
  bulkFeedback: string;
  assessment: string;
}

export interface accountDashboardResponse {
  dataList: TableData[];
  loading: boolean;
}
