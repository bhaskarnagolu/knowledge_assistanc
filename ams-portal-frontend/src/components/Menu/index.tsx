import React from 'react';
import { Menu, MenuButton, MenuList, MenuItem, Text } from "@chakra-ui/react";
import Image from 'next/image';
import { IMenuLists } from '@/models';

interface IDropdownMenuProps {
  svgIcon: string;
  menuLists: IMenuLists[];
}

export default function DropdownMenu({
  menuLists,
  svgIcon,
}: IDropdownMenuProps) {
  return (
    <Menu>
      <MenuButton>
        <Image src={svgIcon} alt={svgIcon} width={20} height={20} priority />
      </MenuButton>
      <MenuList>
        {menuLists.map((el: IMenuLists, idx: number) => (
          <MenuItem key={idx} color="black">
            {el.icon}
            <Text pl={3}>{el.title}</Text>
          </MenuItem>
        ))}
      </MenuList>
    </Menu>
  );
}
