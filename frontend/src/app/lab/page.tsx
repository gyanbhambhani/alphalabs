'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent } from '@/components/ui/card';

export default function LabRedirect() {
  const router = useRouter();

  useEffect(() => {
    router.replace('/research');
  }, [router]);

  return (
    <div className="flex items-center justify-center h-64">
      <Card>
        <CardContent className="py-8 text-center">
          <p className="text-muted-foreground">
            Redirecting to Research...
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
