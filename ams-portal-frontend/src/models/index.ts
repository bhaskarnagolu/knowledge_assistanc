import React from 'react';

export interface IModeSelections {
  isSelected?: boolean;
  title: string;
  icon: React.ReactNode;
}

export interface IReasons {
  title: string;
  isSelected?: boolean;
}

export interface IMenuLists {
  icon?: React.ReactNode;
  title: string;
}
