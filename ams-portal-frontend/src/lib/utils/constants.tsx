import {
  IMenuLists,
  IModeSelections,
  IReasons,
} from '@/models';
import Image from 'next/image';

const reasons: IReasons[] = [
  {
    title: 'Not correct',
    isSelected: false
  },
  {
    title: 'Not relevant',
    isSelected: false
  },
  {
    title: 'Not enough detail',
    isSelected: false
  },
  {
    title: 'Other (please elaborate)',
    isSelected: false
  },
];

const modeLists: IModeSelections[] = [
  {
    isSelected: true,
    title: 'Brief',
    icon: (
      <Image
        src="/playlist.svg"
        alt="playlist"
        width={20}
        height={20}
        priority
      />
    ),
  },
  {
    title: 'Balanced',
    icon: <Image src="/list.svg" alt="list" width={20} height={20} priority />,
  },
  {
    title: 'Detailed',
    icon: (
      <Image
        src="/expand-categories.svg"
        alt="expand-categories"
        width={20}
        height={20}
        priority
      />
    ),
  },
];

const headerMenuItems: IMenuLists[] = [
  {
    title: 'Rename',
  },
  {
    title: 'Delete',
  },
];

const chatFilterMenuItems: IMenuLists[] = [
  {
    icon: <Image src="/list.svg" alt="list" width={25} height={20} priority />,
    title: 'Shorter',
  },
  {
    icon: (
      <Image
        src="/expand-categories.svg"
        alt="expand-categories"
        width={25}
        height={20}
        priority
      />
    ),
    title: 'Longer',
  },
  {
    icon: (
      <Image
        src="/portfolio.svg"
        alt="portfolio"
        width={25}
        height={20}
        priority
      />
    ),
    title: 'More professional',
  },
  {
    icon: (
      <Image
        src="/palm-tree.svg"
        alt="palm-tree"
        width={25}
        height={20}
        priority
      />
    ),
    title: 'More casual',
  },
];

export { modeLists, reasons, headerMenuItems, chatFilterMenuItems };
