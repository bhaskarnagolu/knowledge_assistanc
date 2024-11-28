import { Box, Center, chakra, useRadio } from "@chakra-ui/react"
import { Image, Text } from "@chakra-ui/react"

const CustomRadio = (props: { [x: string]: any; image: string, alttext: string }) => {
  const { image, alttext, align, ...radioProps } = props
  const { state, getInputProps, getRadioProps, htmlProps, getLabelProps } =
    useRadio(radioProps)

  return (
    <>
      <chakra.label {...htmlProps} cursor='pointer'>
        <input {...getInputProps({})} hidden />
        <Box
          {...getRadioProps()}
          bg={state.isChecked ? 'green.200' : 'transparent'}
          w={12}
          p={1}
          rounded='full'
          width={10}
        >
          <Image src={image} rounded='full' {...getLabelProps()} alt={alttext}/>
        </Box>
      </chakra.label>
      <Text textAlign={align ?? 'center'}>{alttext}</Text>
      <Text></Text>
      <Text></Text>
    </>
  )
}

export default CustomRadio;