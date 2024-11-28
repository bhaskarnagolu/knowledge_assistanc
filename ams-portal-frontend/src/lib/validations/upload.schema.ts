import { z } from "zod";

export const FILE_KEY = 'uploadFile';
export const ACCEPTED_FILE_TYPES = ["text/csv"];
export const ACCEPTED_INPUT_FILES = ["text/csv", "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"];

export const UploadFormSchema = z.object({
  [FILE_KEY]: z.custom<File>(val => val instanceof File, "Please upload a file")
});

export type UploadFormInput = z.infer<typeof UploadFormSchema>;
