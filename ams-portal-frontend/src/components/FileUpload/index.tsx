import { FormErrorMessage, useToast } from "@chakra-ui/react";
import { useFormContext, Controller } from "react-hook-form";
import { ACCEPTED_INPUT_FILES } from "@/lib/validations/upload.schema";
import { useEffect } from "react";

type FileUploadProps = {
  name: string;
  accept?: string;
  multiple?: boolean;
  files: File[];
  setFiles?: any;
}

const FileUpload = (props: FileUploadProps) => {
  const toast = useToast();
  const filesList = [] as File[];

  function handleFile(targetFiles: FileList | null) {
    if(targetFiles!=null) {
      for (let i = 0; i < targetFiles.length; i++) {
        const fileType = targetFiles[i]['type'];
        if (ACCEPTED_INPUT_FILES.includes(fileType)) {
          filesList.push(targetFiles[i]);
        } else {
          toast({
            title: "only .xlsx/.csv files are accepted",
            status: "error",
            variant: "left-accent",
            duration: 2000
          });
        }
      }
    }
    //invoke the parent's callback handler function and pass data to it
    props.setFiles(filesList);
  }

  const { name, accept, multiple } = props;
  const {
    control,
    formState: { errors },
  } = useFormContext();

  return (
    <>
      <Controller
        name={name}
        control={control}
        render={({ field: { ref, name, onBlur, onChange } }) => (
          <input
            type="file"
            ref={ref}
            name={name}
            multiple={multiple || false}
            accept={accept}
            onBlur={onBlur}
            onChange={(e) => {handleFile(e.target.files); onChange(e.target.files?.[0])}}
          />
        )}
      />
      {errors[name] && (
        <FormErrorMessage>
          {errors[name]?.message as string}
        </FormErrorMessage>
      )}
    </>
  )
}

export default FileUpload;